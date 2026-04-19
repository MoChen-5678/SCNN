# SCNN - 核物理波函数神经网络求解器

基于深度学习的相对论平均场理论（RMF）核子波函数求解器，使用**物理信息神经网络（PINN）**结合谱卷积与 GRU 学习满足径向狄拉克方程的束缚态波函数。

## 项目概述

本项目实现了一个物理信息神经网络（PINN），用于求解原子核中核子的径向狄拉克方程。核心创新：

1. **相对论多尺度重整化**：F分量（小分量）天然幅度为G的v/c≈0.05倍，通过架构层尺度注入(*0.05)和残差空间均衡(200x)解决梯度消失问题
2. **Wang et al. 2025 5PADF差分方案**：G/F分量交替使用前向/后向非对称差分，保证Dirac Hamiltonian厄米性，消除虚假态
3. **Rayleigh商梯度阻断**：能量与波函数的Master-Slave锁定机制，打破"左脚踩右脚"不稳定性
4. **FD矩阵全局缓存**：消除Python for循环的O(N³)瓶颈，训练速度提升10~50倍

## 物理背景

### 径向狄拉克方程

根据相对论平均场理论（RMF），核子在原子核中的运动由径向狄拉克方程描述：

```
G' = -(κ/r)G + (ε+2M-Σ_-)F    (大分量方程)
F' = +(κ/r)F - (ε-Σ_+)G        (小分量方程)
```

其中：
- G(r), F(r): 大分量和小分量径向波函数（F为小分量，|F|/|G|≈v/c≈0.05）
- κ: 相对论量子数
- Σ_±: 标量和矢量自洽势（fm⁻¹）
- M: 核子质量（939 MeV/c²）
- ε: 单粒子结合能（E_total - M，已扣除静止质量）

### 关键物理约束

**Rayleigh商能量计算**（量纲统一到MeV）：
```
ε = <ψ|h|ψ> / <ψ|ψ>
hψ_g = ħc·[-F' + (κ/r)F + vps·G]   (注意：所有fm⁻¹项乘hbc=197.33 MeV·fm转换)
hψ_f = ħc·[+G' + (κ/r)G + vms·F]
```

**5PADF差分方案**（Wang et al., Chin. Phys. C 49, 014106, 2025）：
- G分量：前向差分（前向5PADF在左边界，后向4PADF在右边界）
- F分量：后向差分（与G相反，保证厄米性）
- 精度：O(dr⁴)

## 项目结构

```
SCNN/
├── Data_Loader.py             # 数据加载与预处理（按核素分组采样）
├── Model_Architecture.py      # 神经网络模型（FNO+GRU+轨道自注意力）
├── Physics_Informed_Loss.py    # 物理约束损失函数（6项核心损失）
├── Train.py                   # 训练主程序（3阶段课程学习）
├── evaluate_final.py          # 模型评估脚本
├── RHFConsistencyChecker.py   # RHF一致性实时诊断
├── plots/                     # 训练可视化输出
└── checkpoints/               # 模型检查点
```

## 安装依赖

```bash
pip install torch numpy matplotlib pandas
```

## 数据准备

数据应放在 `/home/ubuntu/rhf/results` 目录下：

```
results/
├── 16O/
│   ├── WAV/               # 波函数数据
│   │   ├── 1s1-2.dat      # 1s1/2 轨道
│   │   └── ...
│   └── POT/               # 势场数据
├── 40Ca/
└── ...
```

每个 `.dat` 文件包含：r, G, F, V_ps, V_ms 等列。

## 快速开始

```bash
cd SCNN
python Train.py
```

训练完成后评估：

```bash
python evaluate_final.py
```

## 核心组件说明

### Model_Architecture.py

定义了 `RHF_FNO_GRU` 模型，结合：
- **FNO (Fourier Neural Operator)**：处理径向网格上的场量
- **GRU (Gated Recurrent Unit)**：处理序列收敛历史
- **轨道自注意力 (OrbitalSelfAttention)**：模拟DFT密度求和
- **条件编码**：嵌入量子数κ、主量子数n、质子/中子标识

**关键架构决策（v8相对论尺度注入）**：
```python
# Sobolev平滑后，注入F分量的物理尺度
raw_f = gf_smoothed[:, 1, :] * 0.05   # F是"小分量"，物理幅度≈G×0.05
alpha_f = alpha_g                         # G和F共用相同衰减指数（渐近行为）
```

### Physics_Informed_Loss.py

**6项核心损失函数**：

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `loss_pde` | 10.0 | 狄拉克方程残差（5PADF差分），**Rf加权200x补偿尺度悬殊** |
| `loss_norm` | 2.0 | 归一化约束 ∫(G²+F²)dr = 1 |
| `loss_node` | 8.0 | 径向节点数约束 |
| `loss_smoothness` | 0.5 | Sobolev平滑正则化 |
| `loss_tail` | 1.0 | 远场尾部衰减约束 |
| `loss_energy_rayleigh` | 5.0 | **梯度阻断的Master-Slave能量锁定** |

**关键修复（v9）**：
```python
# FD矩阵全局缓存：相同(n,dr,direction)只构建一次
get_cached_fd_matrix_5padf(n, dr, direction, device)

# Rayleigh梯度阻断：能量读数器detach()，阻止误差回传波函数
loss_energy_rayleigh = torch.mean((E_net - E_rayleigh.detach()) ** 2)

# 训练期禁用相位翻转：PDE对全局符号免疫，自然坍缩
# g_aligned, f_aligned = _align_phase(g, f)  # 已禁用
```

### Train.py

**3阶段课程学习策略**：

| 阶段 | Epoch | 核素 | 态数 |
|------|-------|------|------|
| Phase 1 | 1-500 | 16O, 40Ca (2个) | 7个核心束缚态 |
| Phase 2 | 501-1200 | 16O, 40Ca (2个) | 15态 |
| Phase 3 | 1201-3700 | 全部核素 | 42态 |

## 物理修复记录（v5→v9）

### v5: 量纲灾难修复
- **问题**：Rayleigh商中M_nucleon(939MeV)与fm⁻¹量纲直接相加
- **修复**：移除M_nucleon，所有fm⁻¹项乘hbc(=197.33 MeV·fm)转MeV
- **验证**：Rayleigh能量从~900MeV回落至~100MeV合理范围

### v6: F分量反坍缩
- **问题**：alpha_f=1.97*alpha_g违反Dirac渐近行为；优化器将F→0以规避动能惩罚
- **修复**：alpha_f=alpha_g（相同衰减率）；添加loss_f_dominance强制F²≥0.3G²

### v7: 2Mc²静止质量（已撤销）
- **说明**：用户指出训练数据已使用结合能ε=E_total-M，无需补回2M
- **状态**：完全回退

### v8: 相对论多尺度重整化
- **问题**：网络输出O(1)，但F物理幅度O(0.05)；MSE天然偏好G导致F被忽略
- **修复**：
  1. 架构层：`raw_f *= 0.05` 注入v/c先验尺度
  2. 残差层：Rf权重3.0→200.0补偿梯度悬殊

### v9: 三重致命谬误修复
1. **FD矩阵O(N³)瓶颈**：全局缓存，6处调用全走缓存，速度提升10~50x
2. **能量-波函数死循环**：Rayleigh loss用`.detach()`阻断，打破Moving Target不稳定性
3. **相位拓扑翻转**：训练期禁用_align_phase，推理阶段再对齐

## 输出与可视化

训练过程中自动生成：

1. **波函数对比图** (`plots/wavefunction_1s12_epoch*.png`):
   - 预测 vs 真实 G(r) 和 F(r)
   - 径向概率密度 ρ(r) = G² + F²
   - 逐点绝对误差

2. **训练日志** (`train_log_v*.txt`):
   - 各损失分量随epoch变化
   - E_rayleigh, E_network, E_kin, norm等诊断量

3. **模型检查点** (`checkpoints/`):
   - 每150 epoch保存

## 参考文献

1. Wang et al., Chin. Phys. C 49, 014106 (2025) — 5PADF方案
2. Ring P, Schuck P. The Nuclear Many-Body Problem. Springer, 2004.
3. Greiner W. Quantum Mechanics: Symmetries. Springer, 1994.

## 更新日志

### v9 (2026-04-19)
- **修复**: FD矩阵全局缓存，训练速度10~50x提升
- **修复**: Rayleigh梯度阻断，消除能量-波函数死循环
- **修复**: 训练期禁用相位翻转，避免拓扑翻转

### v8 (2026-04-19)
- **修复**: 相对论多尺度重整化，架构层F尺度注入+残差200x均衡

### v7 (2026-04-19)
- **修复**: 2Mc²补回（已撤销）

### v6 (2026-04-19)
- **修复**: alpha_f=alpha_g统一衰减率；loss_f_dominance防F归零

### v5 (2026-04-19)
- **修复**: Rayleigh商量纲统一；真空边界条件整体掩码

### v2.0 (2025-04-18)
- **新增**: 轨道自注意力机制
- **新增**: 全核素训练支持（37个核素）
- **优化**: 损失函数精简（12项→6项）

## 作者

- 项目维护者：MoChen
- 基于 CodeBuddy AI 辅助开发

## 许可证

MIT License
