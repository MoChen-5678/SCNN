# DPL-RMF: 神经网络 RMF 变分求解器

本目录主线实现 PRL/AI2DFT 风格的核物理变分 DFT。默认先无网络直接参数化独立物理分量
`S,V0,V3,VC`，严格组装局域 Dirac Hamiltonian 四通道，再以 `E[H_theta]` 和重构项
`H_theta - H_tilde` 的广义梯度训练。默认 `torch-rmf` 后端在同一个 PyTorch 计算图中完成
fixed-H Dirac 求解、密度、RMF 场重构、无质心能量和 `d(E/A)/dH`。

首版只支持 `IE=1` 的 RMF 参数组：`DD-ME1`、`DD-ME2`、`PKDD`、`TW99`、`DD-LZ1`。
`PKA1/PKO*` 属于 RHF/Fock 范围，保留到第二版。

## 文档依据

- `核物理(1).pdf`：RMF/RHF 自洽流程、球形有限核径向 Dirac 求解、密度和平均场重建。
- `RHF-e.pdf`：相对论 Hartree-Fock 形式、能量泛函和通道分解。
- `PhysRevLett.133.076401(1).pdf`：AI²DFT 的变分能量最小化和神经网络 DFT 思想。
- `基于神经网络DFT变分求解器替换核物理自洽迭代的可行性研究.pdf`：推荐先替换外层迭代而不是直接替换物理内核。

## 文件结构

- `dpl_rhf/cli/prl_train.py`：新版 PRL Hamiltonian 模式入口。
- `dpl_rhf/backends/torch_rmf.py`：默认可微 RMF 后端，负责 Dirac 求解、能量和 autograd 梯度。
- `dpl_rhf/backends/fortran_rmf.py`：Fortran fixed-H 诊断后端，保留用于对比。
- `dpl_rhf/models/hamiltonian_net.py`：只输出局域 Hamiltonian 的 SiLU 网络。
- `dpl_rhf/functionals/pkdd_action.py`：严格 PKDD-RMF 作用量、密度、场和能量。
- `dpl_rhf/functionals/pkdd_rmf.py`：PKDD/RMF 通道定义和初始场。
- `dpl_rhf/legacy/`：不受支持的历史实现，仅保留兼容导入和外部对比工具。
- `docs/PRL_RMF_DESIGN.md`：新版设计说明。
- `docs/STRICT_PRL_RMF_MIGRATION.md`：旧实现替换项和数值验收记录。
- `USAGE.md`：参数和输出文件说明。

## 编译 Core-1204

Python 依赖：

```bash
python -m pip install -r requirements.txt
```

桥接接口位于上级目录 `../Core-1204`。首次运行或修改 Fortran 后需要重建：

```bash
cd ../Core-1204
make -f Makefile.build
cd ../DFT-RHF
```

## 快速检查

```bash
python dpl_rmf.py inspect-core
```

运行传统 RMF SCF 基线：

```bash
python dpl_rmf.py scf-baseline --model DD-ME2 --z 8 --n 8 --out outputs/o16_scf
```

运行 PRL/AI2DFT Hamiltonian 模式：

```bash
python -m dpl_rhf.cli.prl_train --model PKDD --z 8 --n 8 \
  --backend torch-rmf --mode direct --epochs 500 --lambda-reconstruct 0.001 \
  --energy-gradient-weight 1.0 --derivative-order 7 --device cuda \
  --compare-scf --out outputs/prl_pkdd_O16
```

## 方法概述

每个 PRL Hamiltonian epoch 执行：

1. 直接系数或 SiLU 网络输出 `S,V0,V3,VC`，硬约束层组装 `H_theta(r)`。
2. `torch-rmf` 后端用 7 点非中心差分构造 Hermitian Dirac 矩阵并求解 occupied no-sea 分支。
3. 后端由占据态密度重构 RMF 势 `H_tilde`，计算书中无质心 RMF 能量。
4. Python 训练器组合 PRL 广义梯度 `grad_E_H + lambda(H_theta-H_tilde)`。
5. `grad_E_H` 由 `torch.autograd.grad(E/A, H_theta)` 得到，再用 surrogate loss 反传到网络参数。

训练不使用 SCF 能量或势场作为标签。Fortran SCF 只在 `--compare-scf` 时作为训练后的外部盲测。
直接变分的内外物理门未通过时，程序拒绝超过 200 轮的网络训练。

## 输出

每次运行在 `--out` 目录写出：

- `history.csv`：迭代历史、残差和能量。
- `summary.json`：最终能量、收敛状态和运行参数。
- `dpl_hamiltonian.npz`：网络输出的局域 Hamiltonian。
- `reconstructed_hamiltonian.npz`：后端密度重构的 `H_tilde`。
- `autograd_gradient.npz`、`gradient_check.json`：可微 RMF 梯度及有限差分校验。
- `formula_invariants.json`：Hermiticity、作用量、谱、节点、度量和梯度总验收。
- `discrete_metric_check.json`：轨道矩阵元与密度积分的离散恒等式。
- `channel_gradient_projection.csv`：四个物理分量中的 PRL 梯度投影。
- `density_diagnostics.npz`、`field_diagnostics.npz`：后端诊断。
- `model.pt`：Hamiltonian 网络权重。

## 第二版范围

RHF 替换需要额外处理非局域 Fock 势 `epotl`、`ρ-T`、`ρ-VT`、`π-PV`、张量通道和交换能量残差。当前 `dpl_rhf/backends/fortran_rhf_stub.py` 只保留接口并明确报错，避免把 RMF DPL 误用于 RHF 物理。
