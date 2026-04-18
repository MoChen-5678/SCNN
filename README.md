# SCNN - 核物理波函数神经网络求解器

基于深度学习的相对论平均场理论（RMF）核子波函数求解器，使用谱卷积神经网络（Spectral CNN）结合**轨道自注意力机制**学习满足狄拉克方程的束缚态波函数。

## 项目概述

本项目实现了一个物理信息神经网络（Physics-Informed Neural Network），用于求解原子核中核子的径向狄拉克方程。核心创新：

1. **轨道自注意力机制**：模拟DFT密度求和 ρ(r) = Σ_α ν_α ψ_α†ψ_α，让轨道间相互感知
2. **精简物理损失函数**：基于物理约束将12项损失精简为6项核心损失
3. **RHF实时监督**：训练中监控势能-动能-能量一致性，异常自动警告
4. **全核素训练**：支持37个核素（从O到Pb）的联合训练

使网络学习到物理上正确的波函数解。

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
├── Data_Loader.py           # 数据加载与预处理（37核素，分组采样）
├── Model_Architecture.py    # 神经网络模型（含轨道自注意力）
├── Physics_Informed_Loss.py # 物理约束损失函数（精简6项+RHF监督）
├── Train.py                 # 训练主程序（全核素课程学习）
├── evaluate_final.py        # 模型评估脚本
├── backup/                  # 原代码备份
├── plots/                   # 训练可视化输出
└── requirements.txt         # 依赖包列表
```

## 安装依赖

```bash
pip install torch numpy matplotlib pandas
```

## 数据准备

数据应放在 `/home/ubuntu/rhf/results` 目录下，包含以下文件结构：

```
results/
├── 16O/                    # 氧16
│   ├── WAV/               # 波函数数据
│   │   ├── 1s1-2.dat      # 1s1/2 轨道数据
│   │   ├── 1p3-2.dat      # 1p3/2 轨道数据
│   │   └── ...
│   └── POT/               # 势场数据
├── 40Ca/                   # 钙40
│   └── ...
├── 56Ni/                   # 镍56
├── 208Pb/                  # 铅208
└── ...                     # 共37个核素
```

**支持的核素**（37个）：
- 氧同位素：14O, 16O, 18O, 20O, 22O, 24O
- 钙同位素：40Ca, 42Ca, 44Ca, 46Ca, 48Ca, 50Ca, 52Ca
- 镍同位素：56Ni, 58Ni, 60Ni, 62Ni, 64Ni
- 锡同位素：100Sn, 112Sn, 114Sn, 116Sn, 118Sn, 120Sn, 122Sn, 124Sn, 132Sn
- 铅同位素：180Pb, 182Pb, 184Pb, 186Pb, 188Pb, 190Pb, 192Pb, 194Pb, 196Pb, 204Pb, 208Pb

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

训练过程采用**两阶段课程学习策略**：
- **Phase 1** (Epoch 1-300): 轻核课程学习（7个核素：16O, 18O, 20O, 40Ca, 42Ca, 44Ca, 56Ni）
- **Phase 2** (Epoch 301-3000): 全核素训练（37个核素）

所有阶段使用**全部42个轨道态**，通过核素难度递增实现课程学习。

### 2. 评估模型

```bash
python evaluate_final.py
```

## 核心组件说明

### Model_Architecture.py

定义了 `RHF_FNO_GRU` 模型，结合：
- **FNO (Fourier Neural Operator)**: 处理径向网格上的场量
- **GRU (Gated Recurrent Unit)**: 处理序列收敛历史
- **轨道自注意力 (OrbitalSelfAttention)**: 模拟DFT密度求和，让轨道间相互感知
- **条件编码**: 嵌入量子数 κ、主量子数 n、质子/中子标识

**OrbitalSelfAttention** 核心设计：
```python
# Q/K/V 均来自轨道特征，实现真正的自注意力
Q = q_proj(orbital_features)  # (n_orbits, feature_dim)
K = k_proj(orbital_features)
V = v_proj(orbital_features)
# 注意力权重结合占据数 ν_α 作为偏置
attn_scores = Q @ K.T / sqrt(dim) + nu_scale * nu_weights
```

输入：
- X: 初始猜测波函数序列 (B, L, 12, N)
- κ: 相对论量子数 (B,)
- r_grid: 径向网格 (B, N)
- is_proton, z_num, n_num, n_principal: 核素和轨道信息

输出：
- Y: 收敛态波函数 (B, 11, N)，包含 G, F, 势场, 能量等

### Physics_Informed_Loss.py

实现了基于物理约束的**精简损失函数**（12项 → 6项核心损失）：

| 损失项 | 说明 | 来源 |
|--------|------|------|
| `loss_pde` | 狄拉克方程残差，4阶中心差分 | 原 loss_pde |
| `loss_norm` | 归一化约束 ∫(G²+F²)dr = 1 | 原 loss_norm |
| `loss_node` | 径向节点数约束 | 原 loss_node |
| `loss_physical_state` | 正能量 + 动能正定性 | 合并 loss_positive_energy + loss_kinetic_positive |
| `loss_smoothness` | 波形平滑性（形态+边界+尾部） | 合并 loss_shape + loss_boundary_smooth + loss_boundary + loss_tv_far + loss_tail + loss_mono |
| `loss_energy_rayleigh` | Rayleigh商能量一致性 | 原 loss_energy_rayleigh |

**删除冗余项**：loss_amplitude, loss_energy_mse, loss_energy_range, loss_peak

### RHFConsistencyChecker（实时监督）

每N步验证势能-波函数-能量一致性：
```python
checker = RHFConsistencyChecker(check_every=100, lambda_consistency=2.0)
consistency_loss, diagnostics = checker.compute_consistency(pred, kappa, dr)
# 自动检测：E_kin < 0 或 |E_rayleigh - E_network| > 100 MeV 时警告
```

### Train.py

训练主程序，关键超参数：

```python
# 模型配置
use_self_attention = True   # 启用轨道自注意力

# 精简物理损失权重（6项核心损失）
lambda_data = 0.5           # 数据MSE
lambda_pde = 5.0            # PDE残差（主导）
lambda_norm = 5.0           # 归一化
lambda_node = 8.0           # 节点数
lambda_physical = 10.0      # 物理状态（正能量+动能）
lambda_smooth = 5.0         # 平滑性
lambda_rayleigh = 8.0       # Rayleigh商能量一致性
lambda_consistency = 2.0    # RHF一致性监督

# 训练参数
num_epochs = 3000
learning_rate = 1e-4
batch_size = 32

# 课程学习
phase1_isotopes = ['16O', '18O', '20O', '40Ca', '42Ca', '44Ca', '56Ni']  # 7个轻核
phase1_epochs = 300
```

**IsotopeGroupedBatchSampler**: 按核素分组的批采样器，保证同batch内样本来自同一核素（自注意力所需）

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

| 阶段 | Epoch | 核素数量 | 核素列表 | 说明 |
|------|-------|----------|----------|------|
| Phase 1 | 1-300 | 7 | 16O, 18O, 20O, 40Ca, 42Ca, 44Ca, 56Ni | 轻核课程学习 |
| Phase 2 | 301-3000 | 37 | 全部核素（O, Ca, Ni, Sn, Pb） | 全核素联合训练 |

**特点**：
- 所有阶段使用**全部42个轨道态**
- 通过核素难度递增（轻核→重核）实现课程学习
- 自注意力机制让模型学习跨核素的普适物理规律

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

### 2. RHF一致性警告

- 训练初期正常现象，随着训练进行会改善
- 若持续警告，检查 `lambda_consistency` 权重
- 查看 `rhf_consistency_log.csv` 分析异常模式

### 3. 自注意力效果不明显

- 确保使用 `IsotopeGroupedBatchSampler` 分组采样
- 检查同batch内是否来自同一核素
- 验证 `use_self_attention=True` 已启用

### 4. 训练不稳定

- 使用课程学习，从轻核开始
- 调整 `clip_grad_norm`（默认1.5）
- 检查数据质量，排除异常样本
- 观察 `loss_smoothness` 是否收敛

### 5. 多核素训练内存不足

- 减小 `batch_size`
- 使用 `IsotopeGroupedBatchSampler` 减少同时加载的核素
- 考虑使用梯度累积

## 更新日志

### v2.0 (2025-04-18)
- **新增**: 轨道自注意力机制 (`OrbitalSelfAttention`)
- **新增**: 全核素训练支持（37个核素）
- **优化**: 损失函数精简（12项 → 6项核心损失）
- **新增**: RHF一致性实时监督 (`RHFConsistencyChecker`)
- **优化**: 课程学习策略（2阶段，按核素难度递增）

### v1.0
- 初始版本：物理信息神经网络求解RMF方程
- 三阶段课程学习（按轨道态难度）
- 12项物理约束损失函数

## 参考文献

1. 核物理教材第3章：球对称原子核的计算
2. Ring P, Schuck P. The Nuclear Many-Body Problem. Springer, 2004.
3. 相对论平均场理论综述

## 作者

- 项目维护者：MoChen
- 基于 CodeBuddy AI 辅助开发

## 许可证

MIT License
