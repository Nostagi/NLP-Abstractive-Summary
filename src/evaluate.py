from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# Regex mặc định để tách token theo hướng tương thích với ModelConfig.
# Mẫu này giữ lại:
# - số/ngày/giờ có dấu phân cách như 19/4, 10:30, 1.000
# - từ Unicode, phù hợp tiếng Việt có dấu
# - dấu câu/ký hiệu như token riêng nếu xuất hiện trong văn bản
_DEFAULT_TOKEN_PATTERN = re.compile(
    r"""
    \d+(?:[.,:/-]\d+)*
    | [^\W\d_]+(?:[-'][^\W\d_]+)*
    | [^\w\s]
    """,
    re.VERBOSE | re.UNICODE,
)

# Ưu tiên dùng interface thật trong src.interfaces để RougeEvaluator kế thừa đúng thiết kế project.
# Fallback bên dưới chỉ dùng khi chạy riêng evaluate.py trong môi trường chưa cài đủ dependency
# của interfaces.py, ví dụ thiếu torch nhưng vẫn muốn tính ROUGE từ CSV.
try:
    from .interfaces import ModelConfig, evaluate
except ImportError:
    try:
        from interfaces import ModelConfig, evaluate
    except ImportError:

        class ModelConfig:
            split_token_pattern = _DEFAULT_TOKEN_PATTERN

        class evaluate:
            # Giữ cùng tên metric và default max_skip với interface thật.
            rouge_types = (
                "rouge1",
                "rouge2",
                "rougeL",
                "rougeS",
            )

            max_skip = 4


# MetricScore là điểm của một metric đơn lẻ, gồm precision/recall/f1.
# RougeScore là tập điểm nhiều metric: rouge1, rouge2, rougeL, rougeS.
MetricScore = Dict[str, float]
RougeScore = Dict[str, MetricScore]


def _empty_metric() -> MetricScore:
    # Trả về điểm 0 trong các trường hợp không thể tính overlap,
    # ví dụ prediction rỗng, reference rỗng, hoặc không đủ token để tạo n-gram.
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _prf(overlap: int, prediction_total: int, reference_total: int) -> MetricScore:
    # Precision: phần overlap chiếm bao nhiêu trong prediction.
    precision = overlap / prediction_total if prediction_total else 0.0

    # Recall: phần overlap bao phủ bao nhiêu trong reference.
    recall = overlap / reference_total if reference_total else 0.0

    # F1 là trung bình điều hòa của precision và recall.
    # Nếu cả hai đều bằng 0 thì tránh chia cho 0 và trả về 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _overlap_score(
    prediction_units: Iterable[Tuple[str, ...]],
    reference_units: Iterable[Tuple[str, ...]],
) -> MetricScore:
    # Counter giúp tính overlap theo multiset.
    # Ví dụ token "hà" xuất hiện 2 lần thì có thể match tối đa 2 lần.
    prediction_counts = Counter(prediction_units)
    reference_counts = Counter(reference_units)

    prediction_total = sum(prediction_counts.values())
    reference_total = sum(reference_counts.values())

    if prediction_total == 0 or reference_total == 0:
        return _empty_metric()

    # Phép & giữa hai Counter lấy min(count_prediction, count_reference)
    # cho từng đơn vị, đúng với cách ROUGE tính overlap có xét tần suất.
    overlap = sum((prediction_counts & reference_counts).values())
    return _prf(overlap, prediction_total, reference_total)


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    # n-gram là chuỗi n token liên tiếp.
    # ROUGE-1 dùng n=1, ROUGE-2 dùng n=2.
    if n <= 0:
        raise ValueError("n must be positive")

    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _skip_bigrams(tokens: Sequence[str], max_skip: int | None) -> List[Tuple[str, str]]:
    # Skip-bigram là cặp 2 token vẫn giữ thứ tự trái -> phải,
    # nhưng cho phép bỏ qua một số token ở giữa.
    # max_skip=4 nghĩa là giữa hai token được phép có tối đa 4 token bị bỏ qua.
    if len(tokens) < 2:
        return []

    skip_bigrams = []

    for i in range(len(tokens) - 1):
        if max_skip is None:
            right_bound = len(tokens)
        else:
            # +2 vì j là vị trí token thứ hai:
            # j = i + 1 là không skip token nào, j = i + max_skip + 1 là skip max_skip token.
            # right_bound trong range là exclusive nên cần cộng thêm 1 nữa.
            right_bound = min(len(tokens), i + max_skip + 2)

        for j in range(i + 1, right_bound):
            skip_bigrams.append((tokens[i], tokens[j]))

    return skip_bigrams


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    # LCS (Longest Common Subsequence) là dãy con chung dài nhất,
    # cho phép các token không cần nằm liên tiếp nhưng vẫn phải giữ đúng thứ tự.
    # ROUGE-L dùng độ dài LCS làm overlap.
    if not left or not right:
        return 0

    # Dùng dynamic programming với 2 hàng để tiết kiệm bộ nhớ:
    # previous là hàng trước, current là hàng đang tính.
    previous = [0] * (len(right) + 1)

    for left_token in left:
        current = [0] * (len(right) + 1)

        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])

        previous = current

    return previous[-1]


class RougeEvaluator(evaluate):
    # Class triển khai interface evaluate cho bài toán tóm tắt văn bản.
    # reference là bản tóm tắt đúng, prediction là bản model sinh ra hoặc candidate.
    def __init__(
        self,
        lowercase: bool = True,
        max_skip: int | None = None,
        token_pattern: re.Pattern[str] | None = None,
    ):
        if max_skip is not None and max_skip < 0:
            raise ValueError("max_skip must be non-negative or None")

        # lowercase=True giúp giảm khác biệt do viết hoa/viết thường khi so khớp token.
        self.lowercase = lowercase

        # Nếu không truyền max_skip thì dùng default từ interface: 4.
        self.max_skip = self.max_skip if max_skip is None else max_skip

        # Mặc định lấy regex token hóa từ ModelConfig để thống nhất với phần preprocess.
        self.token_pattern = token_pattern or ModelConfig().split_token_pattern

    def tokenize(self, text: str) -> List[str]:
        # Ép None về chuỗi rỗng để evaluator không crash khi dữ liệu CSV có ô thiếu.
        text = "" if text is None else str(text)

        if self.lowercase:
            text = text.lower()

        # findall trả về danh sách token theo regex Unicode.
        return self.token_pattern.findall(text)

    def score_one(self, reference: str, prediction: str) -> RougeScore:
        # Token hóa cả reference và prediction trước khi tính mọi metric.
        reference_tokens = self.tokenize(reference)
        prediction_tokens = self.tokenize(prediction)

        # Tính LCS một lần, sau đó dùng cho ROUGE-L.
        lcs = _lcs_length(prediction_tokens, reference_tokens)

        return {
            # ROUGE-1: overlap unigram giữa prediction và reference.
            "rouge1": _overlap_score(
                _ngrams(prediction_tokens, 1),
                _ngrams(reference_tokens, 1),
            ),
            # ROUGE-2: overlap bigram liên tiếp, nhạy hơn với thứ tự từ cục bộ.
            "rouge2": _overlap_score(
                _ngrams(prediction_tokens, 2),
                _ngrams(reference_tokens, 2),
            ),
            # ROUGE-L: dùng LCS để đo mức giữ đúng thứ tự tổng thể.
            "rougeL": _prf(
                lcs,
                len(prediction_tokens),
                len(reference_tokens),
            ),
            # ROUGE-S: overlap skip-bigram, giữ thứ tự token nhưng cho phép bỏ qua token ở giữa.
            "rougeS": _overlap_score(
                _skip_bigrams(prediction_tokens, self.max_skip),
                _skip_bigrams(reference_tokens, self.max_skip),
            ),
        }

    def score_batch(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> RougeScore:
        # Mỗi reference phải đi cùng đúng một prediction.
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length")

        # Batch rỗng vẫn trả đúng cấu trúc metric để code gọi phía ngoài không bị lỗi key.
        if len(references) == 0:
            return {rouge_type: _empty_metric() for rouge_type in self.rouge_types}

        # Gom tổng precision/recall/f1 từng metric, sau đó chia trung bình macro theo số mẫu.
        totals = {
            rouge_type: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for rouge_type in self.rouge_types
        }

        for reference, prediction in zip(references, predictions):
            # Tính điểm từng cặp rồi cộng dồn theo metric.
            scores = self.score_one(reference, prediction)

            for rouge_type in self.rouge_types:
                for score_name in ("precision", "recall", "f1"):
                    totals[rouge_type][score_name] += scores[rouge_type][score_name]

        count = len(references)

        # Macro-average: mỗi dòng dữ liệu có trọng số như nhau,
        # không phụ thuộc độ dài article/summary.
        return {
            rouge_type: {
                score_name: value / count
                for score_name, value in metric_scores.items()
            }
            for rouge_type, metric_scores in totals.items()
        }

    def reset(self) -> None:
        # Evaluator hiện không giữ cache hoặc state, nên reset không cần làm gì.
        return None


def flatten_scores(scores: RougeScore) -> Dict[str, float]:
    # Chuyển dict lồng nhau thành dict phẳng để dễ tạo DataFrame:
    # {"rouge1": {"f1": 0.5}} -> {"rouge1_f1": 0.5}
    return {
        f"{rouge_type}_{score_name}": value
        for rouge_type, metric_scores in scores.items()
        for score_name, value in metric_scores.items()
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    reference_col: str,
    prediction_col: str,
    evaluator: RougeEvaluator | None = None,
) -> pd.DataFrame:
    # Kiểm tra tên cột sớm để báo lỗi rõ ràng thay vì KeyError khó đọc hơn.
    if reference_col not in df.columns:
        raise ValueError(f"Missing reference column: {reference_col}")

    if prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column: {prediction_col}")

    evaluator = evaluator or RougeEvaluator()

    # Tính score từng dòng, giữ nguyên thứ tự dòng của DataFrame đầu vào.
    rows = [
        flatten_scores(evaluator.score_one(reference, prediction))
        for reference, prediction in zip(df[reference_col], df[prediction_col])
    ]

    return pd.DataFrame(rows)


def average_scores(scores_df: pd.DataFrame) -> Dict[str, float]:
    # Trung bình các cột metric đã được flatten.
    # Hàm này hữu ích khi muốn tự xử lý DataFrame thay vì dùng score_batch.
    if scores_df.empty:
        return {}

    return scores_df.mean(numeric_only=True).to_dict()


def _print_nested_scores(scores: RougeScore) -> None:
    # In dạng ngắn gọn cho CLI: mỗi metric một dòng với P/R/F1.
    for rouge_type, metric_scores in scores.items():
        print(
            f"{rouge_type}: "
            f"P={metric_scores['precision']:.4f} "
            f"R={metric_scores['recall']:.4f} "
            f"F1={metric_scores['f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    # CLI giúp chạy nhanh evaluator từ terminal trên file CSV.
    # Mặc định prediction-col là article chỉ để smoke test vì train.csv hiện có 2 cột.
    parser = argparse.ArgumentParser(description="Compute ROUGE scores for a CSV file.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--reference-col",
        default="summary",
        help="Column containing reference summaries.",
    )
    parser.add_argument(
        "--prediction-col",
        default="article",
        help="Column containing predictions or candidate summaries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to evaluate from the start of the file.",
    )
    parser.add_argument(
        "--max-skip",
        type=int,
        default=evaluate.max_skip,
        help="Maximum skipped tokens for ROUGE-S skip-bigrams.",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercase normalization before tokenization.",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        default=5,
        help="Number of per-row score rows to print.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Đọc CSV đầu vào. File train.csv hiện có cột article và summary.
    df = pd.read_csv(args.input)

    # limit giúp chạy thử nhanh trên vài dòng đầu trước khi tính toàn bộ dataset.
    if args.limit is not None:
        df = df.head(args.limit)

    evaluator = RougeEvaluator(
        lowercase=not args.no_lowercase,
        max_skip=args.max_skip,
    )

    references = df[args.reference_col].astype(str).tolist()
    predictions = df[args.prediction_col].astype(str).tolist()

    # score_batch dùng để lấy macro-average trực tiếp.
    batch_scores = evaluator.score_batch(references, predictions)

    # evaluate_dataframe dùng để xem điểm từng dòng, tiện debug sample cụ thể.
    row_scores = evaluate_dataframe(
        df=df,
        reference_col=args.reference_col,
        prediction_col=args.prediction_col,
        evaluator=evaluator,
    )

    print("Macro-average ROUGE")
    _print_nested_scores(batch_scores)

    # In một vài dòng đầu để người dùng kiểm tra output có đủ các cột metric.
    if args.show_rows > 0:
        print()
        print(f"First {min(args.show_rows, len(row_scores))} row scores")
        print(row_scores.head(args.show_rows).to_string(index=False))


if __name__ == "__main__":
    main()
