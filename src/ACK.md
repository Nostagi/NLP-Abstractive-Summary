# Bảng tóm tắt các biến thể Transformer

| Phương pháp  | Bài báo                                                 | Ý tưởng chính                                                 | Mức độ áp dụng hiện nay | Ghi chú                                 |
| ------------ | ------------------------------------------------------- | ------------------------------------------------------------- | ----------------------- | --------------------------------------- |
| **Post-LN**  | Attention Is All You Need (2017)                        | LayerNorm đặt sau residual block                              | Thấp                    | Kiến trúc Transformer nguyên bản        |
| **ReZero**   | ReZero is All You Need: Fast Convergence at Large Depth | Thêm hệ số residual học được (\alpha), khởi tạo bằng 0        | Trung bình              | Nghiên cứu optimization rất thú vị      |
| **RMSNorm**  | Root Mean Square Layer Normalization                    | Loại bỏ bước centering của LayerNorm, chỉ chuẩn hóa theo RMS  | Rất cao                 | Được dùng rộng rãi trong LLM hiện đại   |
| **DeepNorm** | DeepNet: Scaling Transformers to 1,000 Layers           | Điều chỉnh residual scaling để huấn luyện Transformer cực sâu | Thấp–Trung bình         | Chủ yếu mang tính nghiên cứu, tham khảo |
| **Decoder-only Transformer**           | Improving Language Understanding by Generative Pre-Training (2018)   | Loại bỏ encoder và cross-attention, chỉ giữ causal self-attention để sinh token tự hồi quy        | Rất cao                 | Gần như là nền tảng của toàn bộ LLM hiện đại      |
| **RoPE (Rotary Positional Embedding)** | RoFormer: Enhanced Transformer with Rotary Position Embedding (2021) | Mã hóa vị trí bằng phép quay trong không gian embedding của Q/K thay vì cộng positional embedding | Rất cao                 | Gần như là chuẩn mặc định hiện nay                |
| **SwiGLU Feed Forward**                | GLU Variants Improve Transformer (2020)                              | Thay FFN dùng ReLU/GELU bằng gated activation kiểu SwiGLU                                         | Rất cao                 | Xuất hiện trong LLaMA, PaLM, Gemma, Qwen, Mistral |



Nếu nhìn từ góc độ năm 2025–2026:

- Post-LN: gần như chỉ còn giá trị lịch sử hoặc tái hiện bài báo gốc.
- RMSNorm: là thứ có giá trị thực tiễn cao nhất trong danh sách.
- ReZero: rất hay nếu bạn nghiên cứu optimization hoặc muốn hiểu vai trò của residual path.
- DeepNorm: đáng đọc để hiểu lý thuyết stability của mạng sâu, nhưng hiếm khi xuất hiện trong các mô hình mã nguồn mở phổ biến.

Nếu từ góc độ thực tiễn 2026:
- Decoder-only + RoPE + RMSNorm + SwiGLU: Code vẫn tương đối đơn giản. Mỗi cải tiến đều có giá trị thực tiễn rõ ràng.
- ReZero và DeepNorm mình sẽ xem là các nhánh nghiên cứu optimization để thử nghiệm sau khi bản nền đã chạy ổn định

---

# Implementation

## 1. Rotary Positional Embedding (RoPE)

### 1.1. Nguyên bản (Absolute) Positional Encoding
Trong kiến trúc Transformer nguyên thủy, thông tin về vị trí tuyệt đối của các token được nhúng bằng các hàm sin và cosin có chu kỳ khác nhau. Vector mã hóa vị trí tuyệt đối $PE$ được tính toán độc lập và **cộng trực tiếp** vào vector biểu diễn từ (Token Embedding) ở ngay đầu vào của mô hình:

$$PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$

$$PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$

$$	V^{Input}_{(pos)} = 	V^{Embedding} + PE_{(pos)}$$

*Trong đó:*
* $pos$ là vị trí tuyệt đối của token trong chuỗi chuỗi văn bản.
* $i$ là chỉ số chiều trong không gian ẩn ($0 \le 2i < d_{model}$).

### 1.2. Rotary Positional Encoding

Thay vì chỉ cộng vào Word Embedding ở đầu vào một lần duy nhất đại lượng phản ánh vị trí tuyệt đối. RoPE đề xuất đưa vị trí tương đối (giữa các từ trong câu với nhau) vào mỗi cặp giá trị Query-Key của từng lớp Attention. 

Với một vector ẩn $x$ (đại diện cho Query hoặc Key) tại vị trí thứ $m$, cấu trúc của ma trận xoay RoPE, ký hiệu $R_{\Theta, m}^d$, được định nghĩa dưới dạng ma trận khối đường chéo:

$$R_{\Theta, m}^d =  \begin{pmatrix} 
        R_1 & 0 & 0 & \dots & 0 \\
        0 & R_2 & 0 & \dots & 0 \\
        0 & 0 & R_3 & \dots & 0 \\
        dots &  dots &  dots & \ddots &  dots \\ 
        0 & 0 & 0 & \dots & R_{d/2} 
\end{pmatrix}$$

Trong đó, mỗi khối $R_i$ là một ma trận xoay 2D tương ứng với một cặp tọa độ ẩn:

$$R_i =  \begin{pmatrix} 
        \cos(m \theta_i) & -\sin(m \theta_i) \\ 
        \sin(m \theta_i) & \cos(m \theta_i) 
\end{pmatrix}$$ 

Trong đó: $$\theta_i = 10000^{-\frac{2(i-1)}{d}}$$

Khi áp dụng lên Query $q_m$ và Key $k_n$:

$$	\tilde{q}_m = R_{\Theta, m}^{d_k} q_m, \quad 	\tilde{k}_n = R_{\Theta, n}^{d_k} k_n$$

Sự kỳ diệu nằm ở phép tính tích vô hướng (Dot Product) của Attention score có thể phản ảnh vị trí tương đối $n-m$:

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = (R_{\Theta, m}^{d_k} q_m)^T (R_{\Theta, n}^{d_k} k_n) = q_m^T \left( R_{\Theta, m}^{d_k} \right)^T R_{\Theta, n}^{d_k} k_n = q_m^T R_{\Theta, n-m}^{d_k} k_n$$

---

## 2. Root Mean Square Normalization (RMSNorm)

### 2.1. LayerNorm gốc
LayerNorm truyền thống chuẩn hóa các kích hoạt ẩn của một vector $x \in \mathbb{R}^d$ dựa trên cả hai đại lượng: giá trị trung bình (Mean - $\mu$) và phương sai (Variance - $\sigma^2$):

$$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$

$$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

$$	\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma +  \eta$$

*Trong đó:*
* $\odot$ là phép nhân từng phần tử (element-wise product).
* $\gamma$ (weight) và $ eta$ (bias) là các tham số có thể học được trong quá trình huấn luyện nhằm khôi phục lại phân phối cần thiết.
* $\epsilon$ là một hằng số rất nhỏ để tránh lỗi chia cho số 0.

Việc tính toán phân phối chuẩn này khá tốn công. Chưa kể một số nghi ngại về việc dịch chuyển theo bias (re-centering) không đóng góp quá lớn vào kết quả tổng thể.

* **Giải pháp từ RMSNorm:** Các tác giả của RMSNorm chứng minh rằng đặc tính ổn định gradient và tăng tốc hội tụ của mạng Transformer đến từ việc điều chỉnh quy mô độ lớn (scaling/variance) của các vector ẩn, chứ không phải từ việc dịch tâm dữ liệu (re-centering) về mức trung bình bằng 0. Do đó, RMSNorm mạnh dạn loại bỏ hoàn toàn bước tính toán giá trị trung bình $\mu$ và tham số bias $ eta$. Thay vì chia cho độ lệch chuẩn, nó chỉ chia vector cho giá trị hiệu dụng bình phương trung bình (Root Mean Square). Điều này giúp giảm chi phí tính toán phần cứng từ 10% đến 50% cho riêng lớp Norm mà không hề làm suy giảm độ chính xác của mô hình.

### 2.2. RMSNorm
Với một vector ẩn $x$, giá trị RMS được tính toán như sau:

$$	\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

Công thức chuẩn hóa tinh gọn của RMSNorm:

$$	\text{RMSNorm}(x) = \frac{x}{	\text{RMS}(x)} \odot \gamma$$

*Trong đó:* Mô hình hoàn toàn lược bỏ tham số dịch vị trí $ eta$ (bias), chỉ giữ lại tham số co giãn $\gamma$ (weight).

---

## 3. SwiGLU Feed Forward

### 3.1. Position-wise FFN
Mạng Feed-Forward truyền thống trong Transformer gồm hai tầng tuyến tính liên tiếp kết hợp với một hàm kích hoạt phi tuyến (thường là ReLU) ở giữa:

$$	\text{FFN}(x) = 	\text{ReLU}(x W_1 + b_1) W_2 + b_2$$

*Trong đó:* 
- $W_1 \in \mathbb{R}^{d_{model} 	\times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} 	\times d_{model}}$ 
- Và lớp ẩn thường được mở rộng kích thước lên gấp 4 lần ($d_{ff} = 4 	\times d_{model}$).


### 3.2. Công thức của module thay thế (SwiGLU)

Xuất phát từ kiến trúc Gated Linear Unit (GLU), luồng dữ liệu đầu vào được chẻ làm đôi song song tạo thành: **Luồng Cổng (Gate)** và **Luồng Giá trị (Value)**. Luồng Cổng sẽ được đưa qua một hàm kích hoạt phi tuyến tính để làm "van" kiểm soát lượng thông tin. Nghiên cứu thực nghiệm năm 2020 của Noam Shazeer cho thấy việc kết hợp GLU với hàm kích hoạt Swish (hay còn gọi là SiLU) đem lại hiệu năng vượt trội nhờ tính chất mượt mà (smoothness) và tính không đơn điệu (non-monotonicity) của Swish. Đồng thời, bias ($b$) cũng được loại bỏ hoàn toàn để tăng hiệu quả lưu trữ. Để giữ số lượng tham số tương đương do kiến trúc mới cần 3 ma trận trọng số thay vì 2, kích thước lớp ẩn được tinh chỉnh thu hẹp từ $4d_{model}$ về mức $\frac{8}{3}d_{model}$.


Hàm kích hoạt Swish (SiLU) được định nghĩa bằng công thức:

$$	\text{Swish}(x) = x \cdot 	\text{sigmoid}(x)$$

Cấu trúc SwiGLU Feed Forward Network nhận ma trận đầu vào $x$ và áp dụng phép nhân ma trận song song thông qua phép nhân từng phần tử $\otimes$:

$$	\text{SwiGLU}(x) = \Big( 	\text{Swish}(x W_1) \otimes (x V) \Big) W_2$$

*Trong đó:*
* Ma trận tạo Luồng Cổng: $W_1 \in \mathbb{R}^{d_{model} 	\times d_{ff\_new}}$
* Ma trận tạo Luồng Giá trị: $V \in \mathbb{R}^{d_{model} 	\times d_{ff\_new}}$
* Ma trận chiếu đầu ra (Down-projection): $W_2 \in \mathbb{R}^{d_{ff\_new} 	\times d_{model}}$
* Tỷ lệ kích thước lớp ẩn tối ưu: $d_{ff\_new} = \lfloor \frac{8}{3}d_{model} \rfloor$

