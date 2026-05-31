from __future__ import annotations

import argparse
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Hai cột chuẩn của bài toán summarization trong project.
TEXT_COL = "article"
SUMMARY_COL = "summary"
PAIR_SEP = " [SEP] "


# Các pattern dùng để audit chất lượng text trước/sau cleaning.
# "mojibake_hint" chỉ là tín hiệu nghi vấn, không phải rule xóa tự động.
PATTERNS = {
    "replacement_char": "\ufffd",
    "mojibake_hint": r"\u00c3|\u00c2|\u00e1\u00bb|\u00e1\u00ba|\u00c6|\u00c4",
    "html_tag": r"<[^>]+>",
    "html_entity": r"&[a-zA-Z]+;",
    "url": r"https?://\S+|www\.\S+",
    "email": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
    "phone_like": r"\b(0|\+84)[0-9\s\.\-]{8,}\b",
    "multi_space": r"\s{2,}",
    "control_char": r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
}
COMPILED_PATTERNS = {
    name: re.compile(pattern)
    for name, pattern in PATTERNS.items()
}


# Regex phục vụ normalize text. Các rule này chỉ xử lý lỗi kỹ thuật,
# không lower-case và không xóa dấu câu để giữ tối đa ngữ nghĩa tiếng Việt.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MULTI_SPACE_RE = re.compile(r"\s+")


VIETNAMESE_ACCENT_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵ"
    r"èéẹẻẽêềếệểễ"
    r"ìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữ"
    r"ỳýỵỷỹ"
    r"đ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    r"ÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨ"
    r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸ"
    r"Đ]"
)


LENGTH_COLS = [
    "article_char_len",
    "summary_char_len",
    "article_space_tokens",
    "summary_space_tokens",
    "compression_ratio_space",
]

WARNING_COLS = [
    "warn_short_article",
    "warn_short_summary",
    "warn_summary_longer",
    "warn_high_compression",
    "warn_low_compression",
]


@dataclass
class PreprocessConfig:
    """Cấu hình đường dẫn và threshold cho pipeline preprocessing."""

    project_root: Path = Path(".")
    train_file: str = "data/train-00000-of-00001.parquet"
    valid_file: str = "data/valid-00000-of-00001.parquet"
    output_dir: str = "data/processed"
    random_state: int = 42
    short_article_tokens: int = 30
    short_summary_tokens: int = 5
    high_compression_ratio: float = 0.8
    low_compression_ratio: float = 0.02

    @property
    def train_path(self) -> Path:
        return self.project_root / self.train_file

    @property
    def valid_path(self) -> Path:
        return self.project_root / self.valid_file

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.output_dir

    @property
    def report_dir(self) -> Path:
        return self.processed_dir / "reports"

    @property
    def figure_dir(self) -> Path:
        return self.processed_dir / "figures"


def ensure_output_dirs(config: PreprocessConfig) -> None:
    """Tạo các thư mục output cho dữ liệu sạch, report và figure."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Đọc một split parquet và kiểm tra hai cột bắt buộc."""
    df = pd.read_parquet(path)
    required_columns = {TEXT_COL, SUMMARY_COL}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{split_name} thiếu cột bắt buộc: {sorted(missing_columns)}")

    # Chỉ giữ schema chính để downstream training nhất quán.
    return df[[TEXT_COL, SUMMARY_COL]].copy()


def basic_overview(df: pd.DataFrame, split_name: str, stage: str) -> Dict[str, int | str]:
    """Tạo thống kê schema cơ bản cho report tổng quan."""
    pair_key = make_pair_key(df)
    return {
        "stage": stage,
        "split": split_name,
        "rows": len(df),
        "columns": len(df.columns),
        "article_null": int(df[TEXT_COL].isna().sum()),
        "summary_null": int(df[SUMMARY_COL].isna().sum()),
        "article_empty": int(df[TEXT_COL].fillna("").astype(str).str.strip().eq("").sum()),
        "summary_empty": int(df[SUMMARY_COL].fillna("").astype(str).str.strip().eq("").sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pairs": int(pair_key.duplicated().sum()),
        "article_equals_summary": int(
            df[TEXT_COL].astype(str).str.strip().eq(df[SUMMARY_COL].astype(str).str.strip()).sum()
        ),
    }


def contains_pattern(value: object, pattern_name: str) -> bool:
    """Kiểm tra một giá trị text có khớp pattern audit hay không."""
    if pd.isna(value):
        return False
    return bool(COMPILED_PATTERNS[pattern_name].search(str(value)))


def text_quality_report(df: pd.DataFrame, split_name: str, stage: str) -> pd.DataFrame:
    """Tạo report đếm lỗi/tín hiệu kỹ thuật theo từng cột text."""
    rows = []
    for col in [TEXT_COL, SUMMARY_COL]:
        for pattern_name in PATTERNS:
            count = int(df[col].map(lambda value: contains_pattern(value, pattern_name)).sum())
            rows.append(
                {
                    "stage": stage,
                    "split": split_name,
                    "column": col,
                    "pattern": pattern_name,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def accent_ratio(text: object) -> float:
    """Tính tỷ lệ ký tự có dấu tiếng Việt, chỉ dùng để audit dữ liệu nghi vấn."""
    text = "" if pd.isna(text) else str(text)
    if not text:
        return 0.0
    return len(VIETNAMESE_ACCENT_RE.findall(text)) / len(text)


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm feature độ dài và quan hệ article-summary cho EDA/audit."""
    df = df.copy()
    df["article_char_len"] = df[TEXT_COL].astype(str).str.len()
    df["summary_char_len"] = df[SUMMARY_COL].astype(str).str.len()
    df["article_space_tokens"] = df[TEXT_COL].astype(str).map(lambda text: len(text.split()))
    df["summary_space_tokens"] = df[SUMMARY_COL].astype(str).map(lambda text: len(text.split()))
    df["compression_ratio_space"] = (
        df["summary_space_tokens"] / df["article_space_tokens"].replace(0, pd.NA)
    )
    df["article_eq_summary"] = (
        df[TEXT_COL].astype(str).str.strip() == df[SUMMARY_COL].astype(str).str.strip()
    )
    df["summary_longer_than_article"] = (
        df["summary_space_tokens"] >= df["article_space_tokens"]
    )
    return df


def length_percentiles(df: pd.DataFrame, split_name: str, stage: str) -> pd.DataFrame:
    """Tạo bảng percentile độ dài để tham khảo khi chọn max length."""
    rows = []
    for metric in LENGTH_COLS:
        rows.append(
            {
                "stage": stage,
                "split": split_name,
                "metric": metric,
                "p50": df[metric].quantile(0.50),
                "p75": df[metric].quantile(0.75),
                "p90": df[metric].quantile(0.90),
                "p95": df[metric].quantile(0.95),
                "p99": df[metric].quantile(0.99),
                "max": df[metric].max(),
            }
        )
    return pd.DataFrame(rows)


def normalize_vietnamese_text(
    text: object,
    replace_url: bool = True,
    replace_email: bool = True,
) -> str:
    """Làm sạch lỗi kỹ thuật nhưng giữ tối đa nội dung gốc tiếng Việt."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entity như &amp; hoặc &quot; nếu dữ liệu crawl còn sót.
    text = html.unescape(text)

    # NFC gom ký tự + dấu về dạng Unicode chuẩn, hữu ích cho tiếng Việt.
    text = unicodedata.normalize("NFC", text)

    # Xóa các ký tự điều khiển hoặc HTML tag khó dùng cho training.
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    # Chuẩn hóa một số whitespace ẩn thường gặp trong dữ liệu web.
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")

    # Chuẩn hóa quote/dash để giảm biến thể ký tự, không xóa dấu câu.
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # URL/email được thay bằng token giữ chỗ để giữ tín hiệu nhưng giảm nhiễu.
    if replace_url:
        text = URL_RE.sub(" <URL> ", text)
    if replace_email:
        text = EMAIL_RE.sub(" <EMAIL> ", text)

    return MULTI_SPACE_RE.sub(" ", text).strip()


def make_pair_key(df: pd.DataFrame) -> pd.Series:
    """Tạo khóa article-summary để tìm duplicate pair."""
    return df[TEXT_COL].astype(str) + PAIR_SEP + df[SUMMARY_COL].astype(str)


def append_remove_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Gắn lý do xóa cho dòng thỏa mask."""
    df.loc[mask, "remove_reason"] = df.loc[mask, "remove_reason"] + f"|{reason}"


def apply_preprocessing(df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean text, thêm cột audit, rồi tách kept/removed cho một split."""
    work_df = df[[TEXT_COL, SUMMARY_COL]].copy()
    work_df["source_index"] = work_df.index
    work_df["split"] = split_name

    # Giữ raw text để audit các dòng bị xóa hoặc bị thay đổi sau cleaning.
    work_df["article_raw"] = work_df[TEXT_COL]
    work_df["summary_raw"] = work_df[SUMMARY_COL]

    work_df[TEXT_COL] = work_df[TEXT_COL].map(normalize_vietnamese_text)
    work_df[SUMMARY_COL] = work_df[SUMMARY_COL].map(normalize_vietnamese_text)
    work_df = add_length_features(work_df)
    work_df["remove_reason"] = ""

    # Chỉ xóa các lỗi chắc chắn không nên đưa vào training.
    append_remove_reason(work_df, work_df[TEXT_COL].str.strip().eq(""), "empty_article")
    append_remove_reason(work_df, work_df[SUMMARY_COL].str.strip().eq(""), "empty_summary")
    append_remove_reason(work_df, work_df["article_eq_summary"], "article_equals_summary")

    work_df["pair_key"] = make_pair_key(work_df)
    append_remove_reason(work_df, work_df.duplicated("pair_key"), "duplicate_pair")

    removed = work_df[work_df["remove_reason"] != ""].copy()
    kept = work_df[work_df["remove_reason"] == ""].copy()
    kept = kept.drop(columns=["pair_key"])
    return kept, removed


def add_warning_flags(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Thêm các cột boolean để review dữ liệu nghi vấn, không dùng để xóa dòng."""
    df = df.copy()

    # Các cột warn_* chỉ sống trong DataFrame audit/report.
    # Khi lưu parquet model-ready, pipeline sẽ drop toàn bộ cột này.
    df["warn_short_article"] = df["article_space_tokens"] < config.short_article_tokens
    df["warn_short_summary"] = df["summary_space_tokens"] < config.short_summary_tokens
    df["warn_summary_longer"] = df["summary_space_tokens"] >= df["article_space_tokens"]
    df["warn_high_compression"] = (
        df["compression_ratio_space"] > config.high_compression_ratio
    )
    df["warn_low_compression"] = (
        df["compression_ratio_space"] < config.low_compression_ratio
    )
    return df


def warning_summary_table(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    """Đếm số dòng bị gắn từng warning flag trong train/valid."""
    table = pd.concat(
        [
            train_df[WARNING_COLS].sum().rename("train").to_frame().T,
            valid_df[WARNING_COLS].sum().rename("valid").to_frame().T,
        ]
    )
    return table.reset_index().rename(columns={"index": "split"})


def build_accent_report(train_df: pd.DataFrame, valid_df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Tóm tắt tỷ lệ ký tự có dấu tiếng Việt theo split/cột."""
    rows = []
    for split_name, df in {"train": train_df, "valid": valid_df}.items():
        for col in [TEXT_COL, SUMMARY_COL]:
            values = df[col].map(accent_ratio)
            rows.append(
                {
                    "stage": stage,
                    "split": split_name,
                    "column": col,
                    "mean": values.mean(),
                    "p50": values.quantile(0.50),
                    "p05": values.quantile(0.05),
                    "min": values.min(),
                    "low_ratio_lt_0_01": int((values < 0.01).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_histogram(df: pd.DataFrame, col: str, title: str, path: Path, bins: int = 100) -> None:
    """Vẽ histogram và lưu vào figures để dùng trong báo cáo EDA."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(train_df: pd.DataFrame, config: PreprocessConfig) -> List[Path]:
    """Lưu các histogram chính của train clean để xem phân phối dữ liệu."""
    outputs = [
        (
            "article_space_tokens",
            "Train Article Length by Space Tokens",
            config.figure_dir / "train_article_space_tokens.png",
        ),
        (
            "summary_space_tokens",
            "Train Summary Length by Space Tokens",
            config.figure_dir / "train_summary_space_tokens.png",
        ),
        (
            "compression_ratio_space",
            "Train Compression Ratio by Space Tokens",
            config.figure_dir / "train_compression_ratio.png",
        ),
    ]
    for col, title, path in outputs:
        plot_histogram(train_df, col, title, path)
    return [path for _, _, path in outputs]


def assert_model_ready_outputs(
    train_model: pd.DataFrame,
    valid_model: pd.DataFrame,
    leakage_report: pd.DataFrame,
    expected_paths: Iterable[Path],
) -> None:
    """Các acceptance check đảm bảo output sạch và đúng schema training."""
    assert list(train_model.columns) == [TEXT_COL, SUMMARY_COL]
    assert list(valid_model.columns) == [TEXT_COL, SUMMARY_COL]

    for split_name, df in {"train": train_model, "valid": valid_model}.items():
        assert df[[TEXT_COL, SUMMARY_COL]].isnull().sum().sum() == 0, f"{split_name} còn null"
        assert df[TEXT_COL].astype(str).str.strip().ne("").all(), f"{split_name} còn article rỗng"
        assert df[SUMMARY_COL].astype(str).str.strip().ne("").all(), f"{split_name} còn summary rỗng"
        assert not add_length_features(df)["article_eq_summary"].any(), f"{split_name} còn article == summary"
        assert make_pair_key(df).duplicated().sum() == 0, f"{split_name} còn duplicate pair"

    overlap_after = len(set(train_model[TEXT_COL]).intersection(set(valid_model[TEXT_COL])))
    assert overlap_after == 0, "Train-valid vẫn còn overlap theo article"
    assert int(leakage_report.loc[0, "valid_rows_removed_by_leakage"]) == 0

    for path in expected_paths:
        assert path.exists(), f"Thiếu output: {path}"


def run_preprocessing(
    config: PreprocessConfig | None = None,
    write_outputs: bool = True,
    make_plot_files: bool = True,
) -> Dict[str, pd.DataFrame | Dict[str, Path]]:
    """Chạy toàn bộ pipeline EDA, cleaning, leakage và ghi output nếu được bật."""
    config = config or PreprocessConfig()
    ensure_output_dirs(config)

    train_raw = load_split(config.train_path, "train")
    valid_raw = load_split(config.valid_path, "valid")

    overview_report = pd.DataFrame(
        [
            basic_overview(train_raw, "train", "raw"),
            basic_overview(valid_raw, "valid", "raw"),
        ]
    )

    train_raw_eda = add_length_features(train_raw)
    valid_raw_eda = add_length_features(valid_raw)

    quality_raw_report = pd.concat(
        [
            text_quality_report(train_raw, "train", "raw"),
            text_quality_report(valid_raw, "valid", "raw"),
        ],
        ignore_index=True,
    )

    accent_raw_report = build_accent_report(train_raw, valid_raw, "raw")

    train_candidate, train_removed = apply_preprocessing(train_raw, "train")
    valid_candidate, valid_removed = apply_preprocessing(valid_raw, "valid")

    train_candidate = add_warning_flags(train_candidate, config)
    valid_candidate = add_warning_flags(valid_candidate, config)
    warning_report = warning_summary_table(train_candidate, valid_candidate)

    train_articles_before = set(train_candidate[TEXT_COL])
    valid_articles_before = set(valid_candidate[TEXT_COL])
    overlap_articles = train_articles_before.intersection(valid_articles_before)

    train_pairs_before = set(make_pair_key(train_candidate))
    valid_pairs_before = set(make_pair_key(valid_candidate))
    overlap_pairs = train_pairs_before.intersection(valid_pairs_before)

    train_leakage_mask = train_candidate[TEXT_COL].isin(overlap_articles)
    train_leakage_removed = train_candidate[train_leakage_mask].copy()
    if not train_leakage_removed.empty:
        train_leakage_removed["remove_reason"] = "|train_valid_article_leakage"

    train_clean = train_candidate[~train_leakage_mask].copy()
    valid_clean = valid_candidate.copy()

    article_overlap_after = len(set(train_clean[TEXT_COL]).intersection(set(valid_clean[TEXT_COL])))
    pair_overlap_after = len(set(make_pair_key(train_clean)).intersection(set(make_pair_key(valid_clean))))

    leakage_report = pd.DataFrame(
        [
            {
                "article_overlap_before": len(overlap_articles),
                "pair_overlap_before": len(overlap_pairs),
                "train_rows_removed_by_article_leakage": int(train_leakage_mask.sum()),
                "valid_rows_removed_by_leakage": 0,
                "article_overlap_after": article_overlap_after,
                "pair_overlap_after": pair_overlap_after,
            }
        ]
    )

    train_model = train_clean[[TEXT_COL, SUMMARY_COL]].copy()
    valid_model = valid_clean[[TEXT_COL, SUMMARY_COL]].copy()

    train_clean_eda = add_length_features(train_model)
    valid_clean_eda = add_length_features(valid_model)

    quality_clean_report = pd.concat(
        [
            text_quality_report(train_model, "train", "clean"),
            text_quality_report(valid_model, "valid", "clean"),
        ],
        ignore_index=True,
    )
    quality_report = pd.concat([quality_raw_report, quality_clean_report], ignore_index=True)

    accent_clean_report = build_accent_report(train_model, valid_model, "clean")
    accent_report = pd.concat([accent_raw_report, accent_clean_report], ignore_index=True)

    length_report = pd.concat(
        [
            length_percentiles(train_raw_eda, "train", "raw"),
            length_percentiles(valid_raw_eda, "valid", "raw"),
            length_percentiles(train_clean_eda, "train", "clean"),
            length_percentiles(valid_clean_eda, "valid", "clean"),
        ],
        ignore_index=True,
    )

    overview_clean_report = pd.DataFrame(
        [
            basic_overview(train_model, "train", "clean"),
            basic_overview(valid_model, "valid", "clean"),
        ]
    )
    overview_report = pd.concat([overview_report, overview_clean_report], ignore_index=True)

    removed_all = pd.concat(
        [train_removed, valid_removed, train_leakage_removed],
        ignore_index=True,
        sort=False,
    )

    output_paths = {
        "train_clean": config.processed_dir / "train_clean.parquet",
        "valid_clean": config.processed_dir / "valid_clean.parquet",
        "overview_report": config.report_dir / "preprocess_overview_report.csv",
        "quality_report": config.report_dir / "preprocess_quality_report.csv",
        "accent_report": config.report_dir / "accent_ratio_report.csv",
        "length_report": config.report_dir / "length_percentiles_space_tokens.csv",
        "warning_report": config.report_dir / "warning_flags_summary.csv",
        "leakage_report": config.report_dir / "leakage_report.csv",
        "removed_rows": config.report_dir / "preprocess_removed_rows.csv",
    }

    figure_paths: List[Path] = []
    if write_outputs:
        train_model.to_parquet(output_paths["train_clean"], index=False)
        valid_model.to_parquet(output_paths["valid_clean"], index=False)
        overview_report.to_csv(output_paths["overview_report"], index=False)
        quality_report.to_csv(output_paths["quality_report"], index=False)
        accent_report.to_csv(output_paths["accent_report"], index=False)
        length_report.to_csv(output_paths["length_report"], index=False)
        warning_report.to_csv(output_paths["warning_report"], index=False)
        leakage_report.to_csv(output_paths["leakage_report"], index=False)
        removed_all.to_csv(output_paths["removed_rows"], index=False)

        if make_plot_files:
            figure_paths = make_figures(train_clean_eda, config)

        expected_paths = list(output_paths.values()) + figure_paths
        assert_model_ready_outputs(train_model, valid_model, leakage_report, expected_paths)

    return {
        "train_raw": train_raw,
        "valid_raw": valid_raw,
        "train_clean": train_clean,
        "valid_clean": valid_clean,
        "train_model": train_model,
        "valid_model": valid_model,
        "overview_report": overview_report,
        "quality_report": quality_report,
        "accent_report": accent_report,
        "length_report": length_report,
        "warning_report": warning_report,
        "leakage_report": leakage_report,
        "removed_rows": removed_all,
        "output_paths": output_paths,
        "figure_paths": {"figures": figure_paths},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese summarization data.")
    parser.add_argument("--project-root", default=".", help="Project root, dùng đường dẫn tương đối mặc định.")
    parser.add_argument("--no-figures", action="store_true", help="Không sinh file histogram trong data/processed/figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(project_root=Path(args.project_root))
    result = run_preprocessing(config=config, write_outputs=True, make_plot_files=not args.no_figures)

    train_model = result["train_model"]
    valid_model = result["valid_model"]
    output_paths = result["output_paths"]
    figure_paths = result["figure_paths"]["figures"]

    print("Preprocessing completed.")
    print("Train clean shape:", train_model.shape)
    print("Valid clean shape:", valid_model.shape)
    print("Saved outputs:")
    for path in output_paths.values():
        print("-", path)
    for path in figure_paths:
        print("-", path)


if __name__ == "__main__":
    main()
