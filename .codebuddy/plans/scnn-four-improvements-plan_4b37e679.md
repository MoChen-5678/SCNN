---
name: scnn-four-improvements-plan
overview: 针对SCNN狄拉克波函数预测网络的四大问题进行系统性改进：(1)完善找峰/相位对齐机制；(2)添加能量势场项输出；(3)添加高斯波包惩罚防网络偷懒；(4)强化边界保护机制抑制数值震荡
design:
  architecture:
    framework: react
    component: shadcn
  fontSystem:
    fontFamily: NA
    heading:
      size: 32px
      weight: 600
    subheading:
      size: 18px
      weight: 500
    body:
      size: 16px
      weight: 400
todos:
  - id: enhance-phase-alignment
    content: 重构 _align_phase 为鲁棒的双阶段找峰机制，统一 Model_Architecture.py 和 Physics_Informed_Loss.py 中的相位约定逻辑
    status: completed
  - id: add-shape-penalty
    content: 在 Physics_Informed_Loss.py 中新增 _waveform_shape_loss 函数实现高斯波包形态惩罚（包络单调性+光滑性+尾部集中性）
    status: completed
    dependencies:
      - enhance-phase-alignment
  - id: add-boundary-protection
    content: 在 Physics_Informed_Loss.py 中新增 _boundary_smoothness_loss 实现多层边界保护（TV惩罚+单调衰减+f整体震荡检测）
    status: completed
    dependencies:
      - enhance-phase-alignment
  - id: integrate-losses
    content: 将新损失项集成到 calc_physics_residual 主函数，更新返回 components dict 含能量E和势场信息
    status: completed
    dependencies:
      - add-shape-penalty
      - add-boundary-protection
  - id: update-train-output
    content: 更新 Train.py：添加新权重参数(lambda_shape/lambda_smooth)、扩展CSV日志含能量/势场、增强绘图含势场子图、更新评估函数
    status: completed
    dependencies:
      - integrate-losses
  - id: update-model-arch
    content: 更新 Model_Architecture.py：统一使用共享找峰函数、增强 ansatz_mask 远场衰减力度
    status: completed
    dependencies:
      - enhance-phase-alignment
---

## Product Overview

针对狄拉克方程相对论平均场(RHF)波函数预测神经网络的四个核心训练问题进行全面改进：相位对齐、能量输出、反偷懒惩罚、边界震荡抑制。

## Core Features

### 问题1: 完善找峰机制，保证相位对齐（第一个峰为正）

- 现有 `_align_phase` 函数（`Physics_Informed_Loss.py` 行247-280）使用 `max_pool1d` 滑窗找局部极大值，但存在以下缺陷：
- 只检查第一个局部极大值的符号，不验证该峰是否为"显著"峰（可能是噪声引起的假峰）
- 与 `Model_Architecture.py` 行442-451 的内区均值符号翻转机制逻辑分散且可能冲突
- 在训练过程中网络输出可能无明确局部极大值（平坦输出），导致 fallback 到全局最大值的不可靠行为
- 改进目标：建立统一、鲁棒的相位约定——找到 g(r) 第一个显著的物理峰，确保其为正值；在 `Model_Architecture.py` 和 `Physics_Informed_Loss.py` 中使用一致的找峰策略

### 问题2: 添加能量势场项输出

- 当前模型输出11通道（g,f,vps,vms,vtt,XG,XF,YG,YF,E,vv），但：
- 能量 E 仅在 PDE 计算中使用（行50），未作为独立监控指标输出
- 势场 vps/vms/vtt 未在任何日志或绘图中展示
- Train.py 的 CSV 日志表头（行490）和评估函数均不含这些物理量
- 改进目标：在日志/绘图/评估中增加能量 E 和关键势场的输出与可视化

### 问题3: 高斯波包形状惩罚机制（防偷懒）

- 现有反偷懒约束（行106-125）：振幅下限 + 方差下限 + 梯度方差约束，但这些仅是统计量约束
- 核心问题：网络可以找到一个满足所有统计量要求但不具有真实波函数形状的"近似解"然后停止优化
- 用户建议：引入类似高斯波包形态的惩罚——如果网络输出不是光滑的单/多峰结构而是任意曲线，施加大权重惩罚
- 改进目标：新增波形形态匹配损失，基于物理波函数的光滑性、单峰/多峰结构、指数衰减尾等特征构建惩罚项

### 问题4: 边界保护机制（禁止震荡）

- r=20fm 处边界问题：
- g 出现数值震荡（非光滑趋零）
- f 直接变成高频震荡波（完全无物理意义）
- 现有边界损失（行193-195）：简单的端点L2惩罚 `g[:,0]² + g[:,-1]²`，无法抑制高频震荡
- ansatz_mask 指数衰减 `exp(-alpha*r)` 在 r=20 处已极小(~0)，理论上应压制远场，但 NN 原始输出 raw_g/raw_f 可能在边界剧烈振荡
- 改进目标：
- 对远场区域（r > 15fm）施加平滑性/单调性约束
- 对 f 整体施加总变差(TV)惩罚防止高频振荡
- 引入边界区域的梯度幅度约束

## Tech Stack

- **框架**: PyTorch (现有项目基础)
- **语言**: Python 3.x
- **核心文件修改范围**: 
- `Physics_Informed_Loss.py` — 主要改动：增强找峰、新增波形形状损失、新增边界保护
- `Train.py` — 配合改动：新增能量/势场输出到日志和绘图、新增损失权重参数
- `Model_Architecture.py` — 小改：统一找峰逻辑、增强 ansatz_mask 边界处理

## Tech Architecture

### 系统架构

本项目采用物理信息神经网络(PINN)架构求解狄拉克方程径向波函数：

```
输入(X: 12ch物理+progress序列) → FNO(条件化傅里叶算子) → GRU(时序演化) 
→ CrossAttention(密度调制) → Decoder → Ansatz整流 → 输出(Y: 11ch物理量)
                                                                    ↓
                                                    Physics_Informed_Loss(多约束损失)
```

### 改动影响链分析

```
Physics_Informed_Loss.py (核心改动)
├── 增强 _align_phase()        ← 统一找峰+相位对齐
├── 新增 _waveform_shape_loss() ← 高斯波包形状惩罚
├── 新增 _boundary_smoothness_loss() ← 边界震荡抑制
├── 增强 calc_physics_residual() 返回值 ← 加入能量E等输出
└── 修改 components dict       ← 新增损失分量

Train.py (配合改动)
├── 新增 lambda_shape, lambda_smooth 权重参数
├── 日志CSV增加能量/势场列
├── plot_wavefunctions 增加势场子图
└── _evaluate 增加能量指标

Model_Architecture.py (小改)
├── 统一找峰逻辑（复用 Physics_Informed_Loss 的实现）
└── 增强 ansatz_mask 远场衰减力度
```

## Implementation Details

### 1. 找峰机制完善（Phase Alignment Enhancement）

**当前问题根因分析**：

- `Physics_Informed_Loss.py` 的 `_align_phase()` 使用固定 order=5 的 max_pool1d 滑窗
- 不区分"显著物理峰"和噪声波动
- `Model_Architecture.py` 用内区均值做符号判断，两套逻辑不一致

**改进方案**：

- **双阶段找峰**：(a) 先用宽松窗口找候选峰 → (b) 验证候选峰是否超过全局最大值的一定比例阈值（如30%），确保是真正的物理峰而非噪声
- **跳过近核区平坦段**：r < 0.5fm 区域 g ≈ 0（由 r^|kappa| 因子决定），在此区域找到的"峰"没有意义。从 r >= 0.5fm 开始搜索
- **统一入口**：在 `Physics_Informed_Loss.py` 中定义 `_find_first_significant_peak()` 作为唯一权威找峰函数，`Model_Architecture.py` 的 forward 中也调用同一策略（通过导入或移动到共享模块）

### 2. 波形形状惩罚（Waveform Shape Penalty / Anti-Lazy Mechanism）

**设计原理**：
物理束缚态波函数的核心形态特征：

1. **有限节点数**：n_nodes 由量子数精确确定（已有约束）
2. **光滑性**：除节点外波形光滑连续，无锯齿/台阶
3. **包络单调性**：|psi(r)| 的包络从峰值向外单调递减（类高斯衰减）
4. **集中性**：概率密度集中在核内区域（r < 8fm），尾部指数衰减

**具体实现 — 高斯波包相似度惩罚**：

```python
# 核心思路：计算预测波函数与"理想高斯型包络"的偏离度
# 对每个样本，根据其主量子数 n 构造参考包络（不一定严格高斯，
# 而是用预测波形自身的峰值位置和宽度构造一个光滑参考）

def _waveform_shape_penalty(g, f, r, dr):
    """
    波形形态惩罚：
    a) 包络单调性：从主峰向外，|psi| 的滑动窗口最大值应单调递减
    b) 光滑性：二阶差分（曲率）不应出现剧烈跳变
    c) 尾部集中性：r > 10fm 区间的概率密度占比应小于阈值
    """
    # a) 包络单调性违反
    prob = g**2 + f**2
    envelope = _compute_envelope(prob, window=10)  # 滑动窗口最大值作为包络
    peak_idx = envelope.argmax(dim=-1)  # 主峰位置
    # 从主峰向右，包络应单调递减
    mono_violation = ...  # ReLU(正向增量) 的累加
    
    # b) 光滑性（二阶导数约束）
    d2g = g[:, 2:] - 2*g[:, 1:-1] + g[:, :-2]
    d2f = f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]
    roughness = torch.mean(d2g**2 + d2f**2)
    
    # c) 尾部能量比例
    tail_start = min(int(10.0 / dr), g.shape[-1])
    tail_frac = prob[:, tail_start:].sum(dim=-1) * dr  # 应该很小
    
    return lambda_mono * mono_violation + lambda_rough * roughness + lambda_tail * tail_fraction_penalty
```

### 3. 边界保护机制（Boundary Oscillation Suppression）

**问题本质**：NN 在 r → R_max 区域缺乏足够的监督信号（数据本身在该区域值很小），导致 NN 可以在此区域"自由发挥"产生振荡。

**多层防护策略**：

```python
def _boundary_protection(g, f, r, dr, r_boundary=15.0):
    """
    远场边界保护（r > r_boundary 区域）：
    
    Layer 1: 总变差(TV)惩罚 — 防止高频锯齿振荡
        TV = sum |x[i+1] - x[i]|, 震荡波TV很大
        
    Layer 2: 单调衰减约束 — 远场|psi|应单调递减（允许微小波动）
        对 r > r_boundary，强制 d|psi|/dr <= epsilon
        
    Layer 3: 曲率上限 — 二阶差分绝对值限制
        防止尖锐转折（震荡的特征）
        
    Layer 4: f 分量整体保护 — f 本身量级小，更容易被噪声淹没
        对全区间 f 施加 TV 约束 + 相对幅度约束
    """
    bidx = int(r_boundary / dr)  # 边界起始索引
    
    # --- Layer 1: 远场 TV 惩罚 ---
    g_far = g[:, bidx:]
    f_far = f[:, bidx:]
    tv_g = torch.abs(g_far[:, 1:] - g_far[:, :-1]).sum(dim=-1).mean()
    tv_f = torch.abs(f_far[:, 1:] - f_far[:, :-1]).sum(dim=-1).mean()
    loss_tv = tv_g + tv_f * 2.0  # f 更严格
    
    # --- Layer 2: 远场单调性（包络递减）---
    g_env_far = torch.max(g_far.abs(), dim=-1)[0]  # 粗略包络
    # 允许最多 2 次方向反转（物理上可能的微弱起伏）
    increments = g_env_far[1:] - g_env_far[:-1]
    positive_jumps = torch.clamp(increments, min=0).sum()
    loss_monotonicity = positive_jumps
    
    # --- Layer 3: 全区间 f 的 TV 保护（f 整体震荡检测）---
    tv_f_full = torch.abs(f[:, 1:] - f[:, :-1]).sum(dim=-1).mean()
    # 参考正常 f 的 TV 上限（经验估计）
    ref_tv_f = torch.mean(torch.abs(f)) * 0.5 * f.shape[-1]  # 粗略估计
    loss_f_oscillation = torch.clamp(tv_f_full - ref_tv_f, min=0)**2
    
    return loss_tv + loss_monotonicity + loss_f_oscillation * 5.0
```

### 4. 能量/势场输出增强

**改动点**：

在 `calc_physics_residual` 的返回 components dict 中增加：

- `'energy_mean'`: E 的均值（标量，MeV 单位）
- `'potential_info'`: 势场在核内区的平均值（用于诊断）

在 `Train.py` 中：

- CSV 日志增加 `energy_pred`, `vps_core`, `vms_core` 列
- `plot_wavefunctions` 增加 3rd row 子图：势场对比（vps, vms, vtt vs r）
- `_evaluate` 增加 energy 相关指标

## Directory Structure

```
/home/ubuntu/rhf/SCNN/
├── Physics_Informed_Loss.py      # [MODIFY] 核心：增强找峰+新增性状损失+边界保护+能量输出
├── Train.py                      # [MODIFY] 配合：新权重参数+日志增强+绘图增强+评估增强
├── Model_Architecture.py         # [MODIFY] 小改：统一找峰+增强ansatz_mask
├── Data_Loader.py                # [UNCHANGED]
└── hyperparameter_search.py      # [UNCHANGED]
```

本任务不涉及创建新的 UI 或大幅重设计现有 UI。这是纯后端深度学习模型的改进任务，主要涉及 Python/PyTorch 代码的逻辑修改和算法增强，无需前端框架或组件库。

### SubAgent

- **code-explorer**
- Purpose: 深入搜索代码库中的相关模式（如找峰逻辑的所有调用点、边界处理的相关代码、能量E的所有引用位置），确保修改方案的完整性
- Expected outcome: 定位所有需要同步修改的代码点，避免遗漏