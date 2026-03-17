import numpy as np
from scipy.interpolate import CubicSpline
import healpy as hp
import os
import matplotlib.pyplot as plt

# --- 参数配置 ---
input_txt = "../earth_1Mhz/B_1M.txt"
output_bin = "B_30M.bin"
target_points = 16384 * 16384 * 12  # 32亿点

def process():
    # 1. 加载并转换为 float32
    print("正在加载原始数据...")
    B = np.loadtxt(input_txt).astype(np.float32)
    n_original = len(B)
    x_original = np.linspace(1, n_original, n_original)

    # 2. 插值计算 (保持 float32 节省内存)
    print("正在进行 Spline 插值 (float32)...")
    cs = CubicSpline(x_original, B)
    x_new = np.linspace(1, n_original, target_points).astype(np.float32)
    
    # 得到单精度结果
    B_interpolated = cs(x_new).astype(np.float32)

    # 3. 保存为 Raw Binary
    print(f"正在保存至 {output_bin}...")
    B_interpolated.tofile(output_bin)
    print(f"完成！文件大小: {os.path.getsize(output_bin)/1e9:.2f} GB")

    # --- 演示：如何从 bin 读取并展示 ---
    print("正在从 bin 文件读取部分数据进行验证展示...")
    # 使用 np.fromfile 读取，必须指定 dtype
    # 如果内存足够，可以全部读取：data_loaded = np.fromfile(output_bin, dtype=np.float32)
    # 如果只想读取一部分展示，可以用 memmap
    data_view = np.memmap(output_bin, dtype='float32', mode='r', shape=(target_points,))
    
    # healpy 展示 (注意：hp.mollview 通常需要特定长度的数组，如 12*nside^2)
    hp.mollview(
        data_view, 
        nest=True, 
        title="Interpolated F32 Image", 
        cmap='viridis'
    )
    plt.savefig("interpolated_30M_image.png")

if __name__ == "__main__":
    process()