import matplotlib.pyplot as plt
import pandas as pd

def LossVisualize(loss_series: pd.Series, 
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
        loss_series.index, 
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
        trend_series.index, 
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