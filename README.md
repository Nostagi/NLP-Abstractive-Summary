# Abstractive Summary

## Giới thiệu:

Project thuộc học phần Xử lý ngôn ngữ tự nhiên HKII_2526_INT3406.

Dựa trên [dataset](./data) cung cấp các văn bản Tiếng Việt từ nhiều nguồn (sách, báo,...) kèm bản tóm tắt ý chính. Thực hiện huấn luyện mô hình LLM chuyên bài toán tóm tắt văn bản trừu tượng (abstractive summary) theo 2 hướng:
1. Xây dựng mô hình Transformer dựa theo bài báo ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) và huẩn luyện from scratch dành cho tác vụ trên.
2. Fine-tune một Pre-trained model bất kỳ (giới hạn dưới 3B) với dataset trên.

## Phương pháp thực hiện:

1. Transformer from scratch:
   
   Chi tiết các bổ sung, sửa đổi đã được implement tổng hợp [tại đây](./src/ACK.md)
   
3. Fine-tuning
   
   Chia làm 2 nhánh fine-tune với kỹ thuật: LoRA và Prefix-tuning.

## Kết quả:

Kết quả được đánh giá dựa trên 2 loại thông số:
1. ROUGE:

    Đánh giá dựa trên tần suất trùng lặp về từ vựng giữa bản tóm tắt mẫu và kết quả do mô hình tạo sinh. Hơi cổ điển và thiên về extractive summary.

2. BERT score:

   Đánh giá bằng cosine similarity sau khi vector hóa 2 bản tóm tắt bằng mô hình thứ ba "bert-base-multilingual-cased". Kiểm tra tương đồng ngữ nghĩa (sematics) tốt hơn, nhưng bị bão hòa về điểm số hơn khi xử lý Tiếng Việt.

Kết quả tổng hợp có thể tham khảo [tại đây](./results/figures)

Phân tích chi tiết hơn cho thấy một số insight:
- Mô hình Transformer được implement vẫn hoạt động gần extractive summary hơn, mặc dù cho thấy đọc hiểu về sematics cơ bản nhưng tỉ lệ copy-paste cao (kéo theo điểm ROUGE cao hơn). Không phân loại được ý chính và các phần phụ chú không liên quan (ví dụ như giới thiệu nhân vật ngoài lề của một số bài báo)
- Mô hình được fine-tune bởi LoRA paraphrase khá nhiều nên về mặt chỉ số hơi thấp hơn.
- Mô hình được fine-tune bởi Prefix-Tuning hơi quá đà, mặc dù diễn đạt uyển chuyển hơn, nhưng cũng có thiên hướng extractive summary.

## Hướng dẫn cài đặt

Nếu chạy local, yêu cầu cài đặt các thư viện trong file: [requirements.txt](./requirements.txt)

Khuyên dùng: venv (Virtual Enviroment)
Hệ điều hành: Windows

```bash
py -m venv .venv                   // [1] Khởi tạo venv cho Windows

.venv\Scripts\activate             // [2] Kích hoạt venv cho Windows

pip install -r requirements.txt    // [3] Cài đặt thư viện cần thiết

```

Sau lần đầu cài đặt, những lần sau chỉ cần kích hoạt [2] lại hoặc IDE sẽ tự kích hoạt thay người dùng.

Tham khảo các notebooks đã upload lên Kaggle [tại đây](./notebooks)

## Contribution
1. [Nguyễn Minh Phúc](https://github.com/minhphuctoiday): fine-tune Qwen 2.5-3B Instruct với Prefix-tuning
2. [Trần Hà Giang](https://github.com/hagiangtran05): fine-tune Qwen 2.5-1.5B Instruct với LoRA
3. [Nguyễn Trường Sơn](https://github.com/Nostagi): cài đặt lại mô hình Transformer
