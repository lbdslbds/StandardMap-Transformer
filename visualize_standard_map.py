# -*- coding: utf-8 -*-
"""
Standard Map Simulation and Phase Space Plotting (Optimized)

This script simulates the Standard Map (Chirikov standard map),
a chaotic dynamical system, and plots its phase space for various
initial conditions. Includes fix for Chinese font display, adjusted colormap,
and fix for legend character display.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib # 导入 matplotlib 以设置字体
import time # 用于计时比较

# --- Matplotlib 字体设置 (解决中文显示问题) ---
# --- Matplotlib Font Setup (to fix Chinese character display issues) ---
try:
    # 尝试设置支持中文的字体 (Try setting fonts that support Chinese)
    # 用户需要确保系统中安装了这些字体之一 (User needs to ensure one of these fonts is installed)
    # 常见的选择有 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei' 等
    # Common choices include 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', etc.
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
    # 解决保存图像时负号'-'显示为方块的问题
    # Fix the display issue for minus sign when saving figures
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("已尝试设置中文字体。如果标签仍显示异常，请确保您的系统已安装 SimHei 或 Microsoft YaHei 等字体。")
    print("Attempted to set Chinese font. If labels still display incorrectly, please ensure fonts like SimHei or Microsoft YaHei are installed on your system.")
except Exception as e:
    print(f"设置中文字体时出错 (Error setting Chinese font): {e}")


def standard_map_optimized(I0, theta0, K, steps):
    """
    使用预分配的 NumPy 数组计算标准映射的轨迹。

    Args:
        I0 (float): 初始作用量值 (Initial action value).
        theta0 (float): 初始角度值 (弧度) (Initial angle value in radians).
        K (float): 混沌参数 (Chaos parameter).
        steps (int): 迭代次数 (Number of iterations).

    Returns:
        tuple: (np.ndarray, np.ndarray) 包含随时间变化的作用量 (I) 和角度 (theta) 值。
               (tuple: (np.ndarray, np.ndarray) containing the action (I) and angle (theta) values over time.)
    """
    # 为提高效率，预分配 NumPy 数组
    # Pre-allocate NumPy arrays for efficiency
    Is = np.zeros(steps + 1)
    thetas = np.zeros(steps + 1)

    # 设置初始条件
    # Set initial conditions
    Is[0] = I0
    thetas[0] = theta0

    # 使用映射方程进行迭代
    # Iterate using the map equations
    I = I0
    theta = theta0
    for i in range(steps):
        # 标准映射方程
        # Standard map equations
        I_new = (I + K * np.sin(theta)) % (2 * np.pi)
        theta_new = (theta + I_new) % (2 * np.pi)

        # 存储新值
        # Store new values
        Is[i+1] = I_new
        thetas[i+1] = theta_new

        # 更新下一次迭代的值
        # Update values for the next iteration
        I, theta = I_new, theta_new

    return Is, thetas

# --- 参数设置 (Parameters) ---
K = 1.5      # 混沌参数 (Chaos parameter / Stochasticity parameter)
steps = 1000 # 迭代步数 (Number of iterations)
num_trajectories = 10 # theta 的初始条件数量 (Number of initial conditions for theta)

# --- 初始条件 (Initial Conditions) ---
# 在 0 和 2*pi 之间生成均匀分布的初始角度
# Generate evenly spaced initial angles between 0 and 2*pi
# 使用 endpoint=False 避免 0 和 2*pi 重复
# Use endpoint=False to avoid duplicating 0 and 2*pi
initial_thetas = np.linspace(0, 2 * np.pi, num_trajectories, endpoint=False)
initial_I = 1.0 # 固定的初始作用量 (Fixed initial action)

# --- 绘图设置 (Plotting Setup) ---
plt.figure(figsize=(12, 7)) # 稍大的图形尺寸 (Slightly larger figure size)
# 使用 'cividis' 颜色映射，通常对比度更好且不那么刺眼
# Use 'cividis' colormap, often has better contrast and is less bright
colors = plt.cm.cividis(np.linspace(0, 1, num_trajectories))

# --- 模拟与绘图 (Simulation and Plotting) ---
print(f"正在运行标准映射模拟 K={K}, steps={steps}...")
print(f"Running Standard Map simulation with K={K} for {steps} steps...")
start_time = time.time()

for i, theta0 in enumerate(initial_thetas):
    # 对每个初始条件运行模拟
    # Run the simulation for each initial condition
    Is, thetas = standard_map_optimized(initial_I, theta0, K, steps)

    # 绘制轨迹点
    # Plot the trajectory points
    # 如果轨迹数量较多，使用颜色映射代替单独的标签以保持清晰
    # Use a colormap instead of individual labels for clarity if num_trajectories is large
    plt.scatter(thetas, Is, s=2, color=colors[i], alpha=0.8) # 稍微增加 alpha 值 (Slightly increased alpha)

end_time = time.time()
print(f"模拟完成，耗时 {end_time - start_time:.2f} 秒。")
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# --- 图形最终调整 (Final Plot Adjustments) ---
# 使用包含中文的标签 (Using labels with Chinese characters)
plt.xlabel('θ (Angle / 角度)', fontsize=14)
plt.ylabel('I (Action / 作用量)', fontsize=14)
plt.title(f'Standard Map Phase Space (K={K}) / 标准映射相空间图', fontsize=16)

plt.xlim(0, 2 * np.pi) # 设置 theta 的范围 (Set limits for theta)
plt.ylim(0, 2 * np.pi) # 根据模运算设置 I 的范围 (Set limits for I based on the modulo operation)
plt.grid(True, linestyle='--', alpha=0.5) # 添加网格线 (Add grid lines with style)

# 如果轨迹数量不多 (<= 10)，则显示图例
# Keep legend if num_trajectories is small, otherwise comment out
if num_trajectories <= 10:
     # 修改图例标签格式，避免使用下标字符 '₀'
     # Modify legend label format to avoid subscript character '₀'
     plt.legend([fr'$\theta_0={th:.2f}$' for th in initial_thetas], loc='upper right', fontsize=9) 

plt.tight_layout() # 调整布局防止标签重叠 (Adjust layout to prevent labels from overlapping)
plt.show() # 显示图形 (Show the plot)
