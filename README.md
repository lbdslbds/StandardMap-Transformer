# StandardMap-Transformer
Transformer 模型预测标准映射系统轨迹
# Transformer 模型预测标准映射系统轨迹

本项目使用基于Transformer的深度学习模型来预测标准映射（Standard Map）系统的轨迹演化。标准映射是一个重要的混沌系统模型，常用于研究哈密顿系统中的混沌现象。

## 项目概述

本项目包含以下主要功能：

1. 生成标准映射系统的模拟数据
2. 使用Transformer模型学习系统动力学
3. 评估模型在单步和多步预测上的表现
4. 可视化预测结果与真实轨迹的对比

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- scikit-learn
- tqdm (可选，用于进度条)

## 安装

1. 克隆本仓库：
```bash
git clone https://github.com/yourusername/standard-map-transformer.git
cd standard-map-transformer
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

直接运行主脚本：
```bash
python standard_map_transformer.py
```

脚本将自动执行以下流程：
1. 生成训练数据
2. 训练Transformer模型
3. 评估模型性能
4. 显示可视化结果

## 主要参数配置

可以在代码中修改以下关键参数：

```python
# 混沌参数
K = 0.2  # Chirikov参数，控制混沌程度

# 模型参数
D_MODEL = 128       # Transformer的隐藏层维度
NHEAD = 8           # 注意力头数
NUM_LAYERS = 4      # Transformer编码器层数
DIM_FEEDFORWARD = 512  # 前馈网络维度
DROPOUT = 0.1       # Dropout率

# 训练参数
EPOCHS = 50         # 训练轮数
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 128    # 批量大小
```

## 项目结构

- `standard_map_transformer.py`: 主脚本，包含数据生成、模型定义、训练和评估的全部代码
- `README.md`: 项目说明文件
- `requirements.txt`: 依赖包列表

## 结果展示

脚本运行后会生成多张可视化图表，包括：

1. 初始训练数据的分布和轨迹示例
2. 训练和验证损失曲线
3. 单条轨迹的预测对比（相位空间和时间序列）
4. 预测误差随时间的变化
![初始数据](https://github.com/user-attachments/assets/2567430a-409f-4ac6-9117-b395958ea60b)
![loss](https://github.com/user-attachments/assets/4ac9f97b-b51a-4bde-a88b-34da133f6f60)
![预测结果](https://github.com/user-attachments/assets/a3b568d9-5794-4b69-9889-1b05b942a143)

## 性能指标

模型评估包括：

1. 单步预测的MSE误差
2. 多步预测的RMSE和相关系数
   - 对I（作用量）和θ（角度）分别评估
   - 考虑角度周期性（模2π）


## 扩展应用

本项目代码可以扩展用于：

1. 研究不同K值下的预测难度
2. 比较不同神经网络架构的性能
3. 探索长期预测的可行性

## 贡献

欢迎提交issue或pull request来改进本项目。

## 许可证

[MIT License](LICENSE)
