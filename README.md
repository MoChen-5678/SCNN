# SCNN - 核物理波函数神经网络求解器

基于深度学习的相对论平均场理论（RMF）核子波函数求解器，使用谱卷积神经网络（Spectral CNN）学习满足狄拉克方程的束缚态波函数。

## 项目概述

本项目实现了一个物理信息神经网络（Physics-Informed Neural Network），用于求解原子核中核子的径向狄拉克方程。核心创新是将核物理的物理约束（狄拉克方程残差、归一化条件、节点数约束等）融入神经网络训练过程，使网络学习到物理上正确的波函数解。

### 物理背景

根据相对论平均场理论（RMF），核子在原子核中的运动由径向狄拉克方程描述：

```
dG/dr + (κ/r)G - [Σ_+(r) + M]F = εG
dF/dr - (κ/r)F + [Σ_-(r) - M]G = εF
```

其中：
- G(r), F(r): 大分量和小分量径向波函数
- κ: 相对论量子数
- Σ_±: 标量和矢量自洽势
- M: 核子质量（939 MeV）
- ε: 单粒子能量（结合能，已扣除静止质量）

## 项目结构

```
SCNN/
├── Data_Loader.py          # 数据加载与预处理
├── Model_Architecture.py   # 神经网络模型定义
├── Physics_Informed_Loss.py # 物理约束损失函数
├── Train.py                # 训练主程序
├── evaluate_final.py       # 模型评估脚本
└── requirements.txt        # 依赖包列表
```

## 安装依赖

```bash
pip install torch numpy matplotlib pandas
```

## 数据准备

数据应放在 `/home/ubuntu/rhf/results` 目录下，包含以下文件结构：

```
results/
├── 16O/
│   ├── it001/           # 中子数据
│   │   ├── 1s1-2.dat    # 1s1/2 轨道数据
│   │   ├── 1p3-2.dat    # 1p3/2 轨道数据
│   │   └── ...
│   └── it002/           # 质子数据
│       └── ...
├── 40Ca/
│   └── ...
└── ...
```

每个 `.dat` 文件包含以下列：
- r: 径向坐标 (fm)
- G: 大分量波函数
- F: 小分量波函数
- V_ps, V_ms: 标量和矢量势
- 其他物理量...

## 快速开始

### 1. 训练模型

```bash
cd SCNN
python Train.py
```

训练过程采用三阶段课程学习策略：
- **Phase 1** (Epoch 1-500): 仅学习核心束缚态（7个态）
- **Phase 2** (Epoch 501-1200): 扩展到中等激发态（21个态）
- **Phase 3** (Epoch 1201-3700): 全轨道精调（42个态）

### 2. 评估模型

```bash
python evaluate_final.py
```

## 核心组件说明

### Model_Architecture.py

定义了 `RHF_FNO_GRU` 模型，结合：
- **FNO (Fourier Neural Operator)**: 处理径向网格上的场量
- **GRU (Gated Recurrent Unit)**: 处理序列收敛历史
- **条件编码**: 嵌入量子数 κ、主量子数 n、质子/中子标识

输入：
- X: 初始猜测波函数序列 (B, L, 12, N)
- κ: 相对论量子数 (B,)
- r_grid: 径向网格 (B, N)
- is_proton, z_num, n_num, n_principal: 核素和轨道信息

输出：
- Y: 收敛态波函数 (B, 11, N)，包含 G, F, 势场, 能量等

### Physics_Informed_Loss.py

实现了基于物理约束的损失函数，包括：

1. **PDE 残差** (`loss_pde`): 狄拉克方程残差，使用4阶中心差分
2. **归一化约束** (`loss_norm`): ∫(G²+F²)dr = 1
3. **节点数约束** (`loss_node`): 根据量子数精确约束径向节点数
4. **能量Rayleigh商** (`loss_energy_rayleigh`): 能量从波函数自然涌现
5. **正能量约束** (`loss_positive_energy`): 标量密度 ρ_s = ∫(G²-F²)dr > 0
6. **动能正定性** (`loss_kinetic_positive`): 纯动能期望值 > 0
7. **能量范围** (`loss_energy_range`): 束缚态能量在 [-80, +50] MeV
8. **波形形态** (`loss_shape`): 类高斯波包结构
9. **峰值位置** (`loss_peak`): 波函数峰值位置匹配
10. **边界平滑性** (`loss_boundary_smooth`): 远场无震荡

### Train.py

训练主程序，关键超参数：

```python
# 物理损失权重
lambda_pde = 5.0           # PDE残差（主导）
lambda_norm = 5.0          # 归一化
lambda_node = 8.0          # 节点数
lambda_energy_rayleigh = 8.0  # Rayleigh商能量一致性
lambda_energy_mse = 0.5    # 能量MSE（大幅降低）
lambda_peak = 10.0         # 峰值位置
lambda_shape = 8.0         # 波形形态

# 训练参数
num_epochs = 3700
learning_rate = 1e-4
batch_size = 32
```

## 物理约束详解

### 能量计算方式

根据教材第3章，单粒子能量通过 **Rayleigh商** 计算：

```
ε = <ψ|h|ψ> / <ψ|ψ> - M

其中 hψ_g = -dF/dr - (κ/r)F + [Σ_+ + M]G
      hψ_f = +dG/dr + (κ/r)G + [Σ_- - M]F
```

关键设计：
- 能量不是网络直接回归的目标
- 能量作为**涌现属性**从波函数通过物理约束计算
- `loss_energy_rayleigh` 确保波函数形状正确时能量自然正确

### 课程学习策略

渐进式增加学习难度，防止Loss爆炸：

| 阶段 | Epoch | 态数量 | 说明 |
|------|-------|--------|------|
| Phase 1 | 1-500 | 7 | 仅核心束缚态（1s1/2, 1p3/2等）|
| Phase 2 | 501-1200 | 21 | 加入第一激发壳层（n=2）|
| Phase 3 | 1201-3700 | 42 | 全部轨道（含高激发态）|

### 数据归一化

- **输入 X**: 使用训练集统计量 (mean, std) 标准化
- **输出 Y**: g/f 通道保持物理单位，其他通道使用 Y 的统计量
- **能量**: 结合能（已扣除939 MeV核子质量），范围约 [-80, 0] MeV

## 输出与可视化

训练过程中自动生成：

1. **波函数对比图** (`plots/wavefunction_1s12_epoch*.png`):
   - 预测 vs 真实 G(r) 和 F(r)
   - 径向概率密度 ρ(r) = G² + F²
   - 逐点绝对误差

2. **训练日志** (`training_logs/training_loss_log.csv`):
   - 各损失分量随epoch变化
   - 学习率、能量、势场等信息

3. **损失曲线** (`training_logs/loss_curves.png`):
   - 总损失、数据损失、PDE损失
   - 学习率调度
   - 课程学习阶段标注

4. **模型检查点** (`checkpoints/`):
   - `rhf_fno_gru_best.pt`: 最佳模型
   - `rhf_fno_gru_epoch*.pt`: 定期保存（每150 epoch）
   - `rhf_fno_gru_final.pt`: 最终模型

## 常见问题

### 1. PDE残差不收敛

- 检查学习率是否过大
- 增加 `lambda_pde` 权重
- 确保数据归一化正确

### 2. 能量MSE收敛但物理不正确

- 这是设计问题，已修复：能量现在通过Rayleigh商从波函数计算
- `lambda_energy_mse` 已降至0.5，主要依靠 `lambda_energy_rayleigh`

### 3. 波函数形状异常

- 检查 `lambda_shape` 和 `lambda_peak` 权重
- 确保节点数约束 `lambda_node` 生效
- 验证边界条件 `lambda_boundary`

### 4. 训练不稳定

- 使用课程学习，从简单态开始
- 调整 `clip_grad_norm`（默认1.5）
- 检查数据质量，排除异常样本

## 参考文献

1. 核物理教材第3章：球对称原子核的计算
2. Ring P, Schuck P. The Nuclear Many-Body Problem. Springer, 2004.
3. 相对论平均场理论综述

## 作者

- 项目维护者：MoChen
- 基于 CodeBuddy AI 辅助开发

## 许可证

MIT License
