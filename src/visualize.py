import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import json
import os
from wordcloud import WordCloud, STOPWORDS

def visualizeLoss(loss_series: pd.Series, 
                  plt_name:str = 'Biểu Đồ Thay Đổi Loss Quá Trình Huấn Luyện', 
                  save_path: str = None, 
                  window_size: int = None):
    """
    Vẽ biểu đồ thay đổi Loss với đường dữ liệu thô làm mờ và đường xu hướng đậm.
    
    Args:
        loss_series (pd.Series): Chuỗi dữ liệu loss, với index là số thứ tự batch.
        save_path (str, optional): Đường dẫn để lưu file ảnh. Mặc định là None.
        window_size (int, optional): Kích thước cửa sổ trượt để tính xu hướng. 
                                     Nếu None, tự động lấy 5% tổng số batch.
        
    Returns:
        fig (matplotlib.figure.Figure): Đối tượng biểu đồ để hiển thị.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. VẼ ĐƯỜNG LOSS THÔ (Màu xanh nhạt, trong suốt)
    ax.plot(
        loss_series.index*200,
        loss_series.values, 
        color='dodgerblue', # Xanh dương nhạt
        alpha=0.3,          # Độ trong suốt (0.0 mờ tịt -> 1.0 rõ nét)
        linewidth=1.0, 
        label='Loss (raw)'
    )
    
    # 2. TÍNH TOÁN VÀ VẼ ĐƯỜNG XU HƯỚNG (Màu xanh đậm)
    # Tự động tính window_size nếu không được cung cấp (mặc định lấy 5% độ dài dữ liệu)
    if window_size is None:
        window_size = max(1, len(loss_series) // 20)
        
    # Tính trung bình trượt (Moving Average)
    # min_periods=1 giúp đường xu hướng được vẽ ngay từ những batch đầu tiên thay vì bị khuyết
    trend_series = loss_series.rolling(window=window_size, min_periods=1).mean()
    
    ax.plot(
        trend_series.index*200, 
        trend_series.values, 
        color='blue',   # Xanh dương đậm
        linewidth=1.5,      # Vẽ nét dày hơn để nổi bật
        label=f'Trend (MA window={window_size})'
    )
    
    # 3. TRANG TRÍ BIỂU ĐỒ
    ax.set_title(plt_name, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Batch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9) # framealpha làm nền legend bớt trong suốt để dễ đọc
    
    fig.tight_layout()
    
    # 4. XỬ LÝ LƯU VÀ ĐÓNG LUỒNG
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {save_path}")
        
    plt.close(fig) 
    
    return fig

def visualizeViolin(model_results: Dict[str, pd.DataFrame], 
                    metric_name: str, 
                    plt_name: str = None, 
                    save_path: str = None):
    """
    Vẽ biểu đồ Violin plot so sánh phân phối điểm số của các mô hình học máy.
    
    Args:
        model_results (Dict[str, pd.DataFrame]): Dictionary ánh xạ từ tên mô hình 
                                                 sang DataFrame kết quả của mô hình đó.
        metric_name (str): Tên cột độ đo cần vẽ (phải tồn tại trong mọi DataFrame).
        plt_name (str, optional): Tên biểu đồ. Nếu None, tự động tạo tên dựa trên metric_name.
        save_path (str, optional): Đường dẫn để lưu file ảnh. Mặc định là None.
        
    Returns:
        fig (matplotlib.figure.Figure): Đối tượng biểu đồ để hiển thị.
    """
    # Ràng buộc số lượng mô hình tối đa là 5
    assert len(model_results) <= 5, f"Chỉ hỗ trợ so sánh tối đa 5 mô hình, hiện tại đang có {len(model_results)}."
    
    data_to_plot = []
    labels = []
    
    # Kiểm tra độ đo và trích xuất dữ liệu
    for model_name, df in model_results.items():
        assert metric_name in df.columns, f"Lỗi: Độ đo '{metric_name}' không tồn tại trong bảng kết quả của mô hình '{model_name}'."
        
        # Loại bỏ các giá trị NaN nếu có để hàm vẽ không bị lỗi
        data = df[metric_name].dropna().values
        data_to_plot.append(data)
        labels.append(model_name)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plt_name is None:
        plt_name = f'Score distribution - Metric: {metric_name.upper()}'
        
    # 1. VẼ VIOLIN PLOT
    # showmeans và showmedians giúp thể hiện rõ các điểm đặc trưng của dữ liệu
    parts = ax.violinplot(data_to_plot, showmeans=True)
    
    # 2. TRANG TRÍ MÀU SẮC CHO TỪNG MÔ HÌNH (Tối đa 5 màu)
    colors = ['dodgerblue', 'lightgreen', 'salmon', 'orchid', 'orange']
    
    # Tô màu phần thân violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    # Chỉnh màu râu (whiskers), vạch max/min, vạch mean, vạch median cho rõ nét
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.2)
    
    # 3. TRANG TRÍ BIỂU ĐỒ (Trục, Grid, Title)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    
    ax.set_title(plt_name, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Score ({metric_name})', fontsize=12, labelpad=10)
    
    # Chỉ bật grid ngang để dễ gióng điểm số
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    fig.tight_layout()
    
    # 4. XỬ LÝ LƯU VÀ ĐÓNG LUỒNG
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ violin plot tại: {save_path}")
        
    plt.close(fig) 
    
    return fig

def visualizeViolinModel(df: pd.DataFrame, 
                         metrics: List[str], 
                         plt_name: str = None, 
                         save_path: str = None):
    """
    Vẽ biểu đồ Violin plot so sánh phân phối điểm số giữa các độ đo khác nhau của cùng một mô hình.
    
    Args:
        df (pd.DataFrame): DataFrame chứa kết quả của một mô hình.
        metrics (List[str]): Danh sách tên các cột độ đo cần vẽ (phải tồn tại trong DataFrame).
        plt_name (str, optional): Tên biểu đồ. Nếu None, tự động tạo tên mặc định.
        save_path (str, optional): Đường dẫn để lưu file ảnh. Mặc định là None.
        
    Returns:
        fig (matplotlib.figure.Figure): Đối tượng biểu đồ để hiển thị.
    """
    # Ràng buộc số lượng độ đo tối đa là 5 để khớp với số lượng màu
    
    data_to_plot = []
    labels = []
    
    # Kiểm tra độ đo và trích xuất dữ liệu
    for metric in metrics:
        assert metric in df.columns, f"Lỗi: Độ đo '{metric}' không tồn tại trong DataFrame."
        
        # Loại bỏ các giá trị NaN nếu có để hàm vẽ không bị lỗi
        data = df[metric].dropna().values
        data_to_plot.append(data)
        labels.append(metric)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plt_name is None:
        plt_name = 'Metric Distribution Comparison'
        
    # 1. VẼ VIOLIN PLOT
    # showmeans và showmedians giúp thể hiện rõ các điểm đặc trưng của dữ liệu
    parts = ax.violinplot(data_to_plot, showmeans=True)
    
    # 2. TRANG TRÍ MÀU SẮC CHO TỪNG ĐỘ ĐO (Tối đa 5 màu)
    colors = ['dodgerblue', 'lightgreen', 'salmon', 'orchid', 'orange']
    
    # Tô màu phần thân violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i%5])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    # Chỉnh màu râu (whiskers), vạch max/min, vạch mean, vạch median cho rõ nét
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.2)
            
    # 3. TRANG TRÍ BIỂU ĐỒ (Trục, Grid, Title)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    
    ax.set_title(plt_name, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metrics', fontsize=12, labelpad=10)
    ax.set_ylabel('Score', fontsize=12, labelpad=10)
    
    # Chỉ bật grid ngang để dễ gióng điểm số
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    fig.tight_layout()
    
    # 4. XỬ LÝ LƯU VÀ ĐÓNG LUỒNG
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ violin plot tại: {save_path}")
        
    plt.close(fig) 
    
    return fig

def summarizeModel(df: pd.DataFrame, 
                   model_name: str, 
                   save_path: str = None) -> str:
    """
    Summarize model performance metrics (mean, std, min, max) and output as JSON.
    
    Args:
        df (pd.DataFrame): DataFrame containing the model's scores across various metrics.
        model_name (str): The name of the model being summarized.
        save_path (str, optional): Path to save the JSON output. Defaults to None.
        
    Returns:
        str: A JSON formatted string containing the summarized statistics.
    """
    # 1. TÍNH TOÁN CÁC THỐNG KÊ CƠ BẢN
    num_records = len(df)
    metrics_summary = {}
    
    # Chỉ lấy các cột có định dạng số (numeric) để tránh lỗi khi tính toán
    numeric_df = df.select_dtypes(include=['number'])
    
    for col in numeric_df.columns:
        # Bỏ qua giá trị NaN khi tính toán để tránh lỗi
        col_data = numeric_df[col].dropna()
        
        # Nếu cột không có dữ liệu hợp lệ sau khi dropna
        if len(col_data) == 0:
            continue
            
        metrics_summary[col] = {
            "mean": f"{float(col_data.mean()):.3f}",
            "std": f"{float(col_data.std()):.3f}" if len(col_data) > 1 else 0.0,
            "min": f"{float(col_data.min()):.3f}",
            "max": f"{float(col_data.max()):.3f}"
        }
        
    # 2. XÂY DỰNG DICTIONARY KẾT QUẢ
    summary_dict = {
        "model_name": model_name,
        "num_records": num_records,
        "metrics": metrics_summary
    }
    
    # 3. FORMAT THÀNH JSON STRING
    json_output = json.dumps(summary_dict, indent=4, ensure_ascii=False)
    
    # 4. LƯU RA FILE NẾU CÓ YÊU CẦU
    if save_path is not None:
        # Đảm bảo thư mục tồn tại trước khi lưu
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"Summary for model '{model_name}' successfully saved to: {save_path}")
        
    return json_output

def visualizeWordCloud(text_series: pd.Series, 
                       plt_name: str = 'Word Cloud Của Tập Dữ Liệu', 
                       save_path: str = None,
                       exclude_words: list = None,
                       phrases_to_keep: list = None):
    """
    Vẽ biểu đồ Word Cloud, hỗ trợ lọc từ nhiễu và giữ nguyên cụm từ tiếng Việt.
    
    Args:
        text_series (pd.Series): Series chứa các đoạn text.
        plt_name (str, optional): Tên biểu đồ.
        save_path (str, optional): Đường dẫn lưu file ảnh.
        exclude_words (list, optional): Danh sách các từ muốn loại bỏ khỏi Cloud.
        phrases_to_keep (list, optional): Danh sách các cụm từ muốn dính liền nhau (VD: ['tóm tắt', 'học máy']).
        
    Returns:
        fig: Đối tượng biểu đồ.
    """
    # 1. TIỀN XỬ LÝ DỮ LIỆU TEXT
    valid_texts = text_series.dropna().astype(str)
    text_corpus = " ".join(valid_texts.values)
    
    # Xử lý dính cụm từ: Thay khoảng trắng bằng dấu gạch dưới (_)
    # Ví dụ: "tóm tắt" -> "tóm_tắt". Lúc này WordCloud sẽ hiểu đây là 1 entity duy nhất.
    if phrases_to_keep:
        for phrase in phrases_to_keep:
            # Chuyển đổi linh hoạt không phân biệt hoa thường
            phrase_nospace = phrase.replace(" ", "_")
            # Cần xử lý cẩn thận hơn bằng regex nếu muốn bám sát ngữ cảnh thực tế, 
            # nhưng replace cơ bản là đủ cho các cụm từ thông dụng.
            text_corpus = text_corpus.replace(phrase, phrase_nospace)
            text_corpus = text_corpus.replace(phrase.title(), phrase_nospace.title())
    
    assert len(text_corpus.strip()) > 0, "Lỗi: Tập dữ liệu text trống."
    
    # 2. XÂY DỰNG TẬP LỌC TỪ (STOPWORDS)
    # Khởi tạo tập stopwords mặc định của thư viện
    stopwords_set = {}
    
    # Thêm các từ người dùng muốn lọc (ép về lowercase để đảm bảo lọc chính xác)
    if exclude_words:
        exclude_words_lower = [word.lower() for word in exclude_words]
        # Nếu cụm từ bị loại bỏ nằm trong phrases_to_keep, ta cũng phải thêm dạng có dấu '_' vào stopwords
        exclude_words_nospace = [word.replace(" ", "_") for word in exclude_words_lower]
        stopwords_set.update(exclude_words_lower + exclude_words_nospace)
        
    # 3. TẠO WORD CLOUD
    wordcloud = WordCloud(
        width=1200, 
        height=800, 
        background_color='white', 
        colormap='viridis',
        max_words=200,
        stopwords=stopwords_set, # Đưa tập từ cần lọc vào đây
        collocations=True,       # Bật True để thư viện tự dò thêm các cụm bigrams tiếng Anh/Việt khác
        contour_width=0
    ).generate(text_corpus)
    
    # 4. VẼ VÀ XUẤT BIỂU ĐỒ
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(plt_name, fontsize=16, fontweight='bold', pad=20)
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ Word Cloud tại: {save_path}")
        
    plt.close(fig) 
    
    return fig