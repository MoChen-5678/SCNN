"""
PINN-RHF 全局配置

物理常数、网格参数、核素参数、SCF迭代参数的集中管理。
所有物理量采用自然单位制（ħ=c=1），能量单位 MeV，长度单位 fm。
"""

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════
#   物理常数
# ═══════════════════════════════════════════════════════════════

HBAR_C = 197.3269804      # ħc (MeV·fm), 用于 fm⁻¹ ↔ MeV 转换
M_NUCLEON = 939.0          # 核子静止质量 (MeV), 近似值
M_PROTON = 938.272         # 质子质量 (MeV)
M_NEUTRON = 939.565        # 中子质量 (MeV)
FM_TO_FM1 = 1.0 / HBAR_C   # MeV → fm⁻¹ 的转换因子

# 电子电荷 (用于库仑势)
E_SQ_OVER_4PI = 1.44e-3    # e²/4πε₀ (MeV·fm), = α·ħc ≈ 1.44e-3 × 197.3 ≈ 0.284? 
                           # 实际: e²/(4πε₀) = 1.439976 MeV·fm (精细结构常数α=1/137)

# ═══════════════════════════════════════════════════════════════
#   网格参数（与 Core-1204 Fortran 一致）
# ═══════════════════════════════════════════════════════════════

DR = 0.10                  # 径向步长 (fm)
R_MIN = 0.0                # 最小半径 (fm), 实际用 dr*0.01 避免 r=0 奇点
R_MAX = 20.0               # 最大半径 (fm)
NPT = int(R_MAX / DR) + 1  # 网格点数 = 201
R_SAFE_OFFSET = DR * 0.01  # r=0 安全偏移, 避免除零

# 预计算径向网格（全局常量）
R_GRID = np.linspace(0, R_MAX, NPT)
R_SAFE = np.maximum(R_GRID, R_SAFE_OFFSET)  # r=0 处替换为小正数

# ═══════════════════════════════════════════════════════════════
#   核素参数配置
# ═══════════════════════════════════════════════════════════════

# 默认核素: ¹⁶O (Z=8, N=8), 与 Fortran 测试一致
DEFAULT_ISOTOPE = '16O'
DEFAULT_Z = 8
DEFAULT_N = 8
DEFAULT_A = DEFAULT_Z + DEFAULT_N  # 16

ISOTOPE_CONFIG = {
    '16O': {'Z': 8,  'N': 8,  'A': 16},
    '40Ca': {'Z': 20, 'N': 20, 'A': 40},
    '208Pb': {'Z': 82, 'N': 126, 'A': 208},
}

# (WS_PARAMS 已删除 — 势场统一从 Shooting POT 文件读取)

# PKA1 参数集 (Phys. Rev. C 76, 034314)
# 直接从 Core-1204/Define.f90 第57-62行转录
PKA1_PARAMS = {
    'IE': 2,                # 2=RHF (含Fock交换), 1=RMF (仅Hartree)
    'rvs': 0.159996,        # 饱和密度 ρ_0 (fm⁻³)
    'amu': [938.9, 938.9],  # 中子/质子质量 (MeV)
    # 介子质量 (MeV)
    'amsig': 488.227904,
    'amome': 783.000000,
    'amrho': 769.000000,
    'ampio': 138.000000,
    # 耦合常数
    'gsig': 8.372672,
    'gome': 11.270457,
    'grho': 3.649857,
    'fpio': 1.030722,
    'grtn': 3.199491,       # ρ-张量耦合
    # 密度依赖参数 (σ)
    'asig': 1.103589,   'bsig': 16.490109,
    'csig': 18.278714,  'dsig': 0.135041,
    # 密度依赖参数 (ω)
    'aome': 1.126166,   'bome': 0.108010,
    'come': 0.141251,   'dome': 1.536183,
    # 密度依赖参数 (同位旋矢量)
    'arho': 0.544017,   'artn': 0.820583,
    'apio': 1.200000,
}

# 同位旋指标 (Fortran约定: it=1→中子, it=2→质子)
TAU_Z = [1.0, -1.0]   # τ_z: 中子=+1, 质子=-1
TAU_C = [0.0, 1.0]    # τ_c: 中子=0, 质子=1

# ¹⁶O 全部 84 态定义（42 中子 + 42 质子，来自 Fortran 参考解）
# name: (kappa, n_principal, degeneracy_2j_plus_1, expected_energy_MeV, is_proton)
# 占据态 (E < 0): 前 3 个中子态 + 前 3 个质子态
# 非占据态 (E > 0): 正能量连续态，仍需求解以构建完整 Fock 矩阵
ALL_STATES_16O = [
    # ═══════════════ 中子 (it=1) ═══════════════
    # N.1s.1/2 ~ N.6s.1/2  (κ=-1, s 态 j=1/2, deg=2)
    ('n-1s1/2', -1, 1, 2, -36.674504, False),
    ('n-2s1/2', -1, 2, 2,  -3.297946, False),
    ('n-3s1/2', -1, 3, 2,   0.970178, False),
    ('n-4s1/2', -1, 4, 2,   3.643438, False),
    ('n-5s1/2', -1, 5, 2,   7.727699, False),
    ('n-6s1/2', -1, 6, 2,  13.057529, False),
    # N.1p.3/2 ~ N.6p.3/2  (κ=-2, p 态 j=3/2, deg=4)
    ('n-1p3/2', -2, 2, 4, -20.109944, False),
    ('n-2p3/2', -2, 3, 4,   1.004756, False),
    ('n-3p3/2', -2, 4, 4,   2.924541, False),
    ('n-4p3/2', -2, 5, 4,   5.999594, False),
    ('n-5p3/2', -2, 6, 4,  10.393622, False),
    ('n-6p3/2', -2, 7, 4,  16.078969, False),
    # N.1d.5/2 ~ N.5d.5/2  (κ=-3, d 态 j=5/2, deg=6)
    ('n-1d5/2', -3, 3, 6,  -5.293774, False),
    ('n-2d5/2', -3, 4, 6,   1.749470, False),
    ('n-3d5/2', -3, 5, 6,   4.480290, False),
    ('n-4d5/2', -3, 6, 6,   8.407623, False),
    ('n-5d5/2', -3, 7, 6,  13.487020, False),
    # N.1f.7/2 ~ N.5f.7/2  (κ=-4, f 态 j=7/2, deg=8)
    ('n-1f7/2', -4, 4, 8,   2.511725, False),
    ('n-2f7/2', -4, 5, 8,   5.170586, False),
    ('n-3f7/2', -4, 6, 8,   7.277895, False),
    ('n-4f7/2', -4, 7, 8,  11.065125, False),
    ('n-5f7/2', -4, 8, 8,  16.581866, False),
    # N.1p.1/2 ~ N.6p.1/2  (κ=+1, p 态 j=1/2, deg=2)
    ('n-1p1/2', +1, 2, 2, -14.054756, False),
    ('n-2p1/2', +1, 3, 2,   1.060148, False),
    ('n-3p1/2', +1, 4, 2,   3.185426, False),
    ('n-4p1/2', +1, 5, 2,   6.460839, False),
    ('n-5p1/2', +1, 6, 2,  10.941847, False),
    ('n-6p1/2', +1, 7, 2,  16.640465, False),
    # N.1d.3/2 ~ N.5d.3/2  (κ=+2, d 态 j=3/2, deg=4)
    ('n-1d3/2', +2, 3, 4,   0.648208, False),
    ('n-2d3/2', +2, 4, 4,   1.915877, False),
    ('n-3d3/2', +2, 5, 4,   4.864144, False),
    ('n-4d3/2', +2, 6, 4,   9.053527, False),
    ('n-5d3/2', +2, 7, 4,  14.397460, False),
    # N.1f.5/2 ~ N.5f.5/2  (κ=+3, f 态 j=5/2, deg=6)
    ('n-1f5/2', +3, 4, 6,   2.520193, False),
    ('n-2f5/2', +3, 5, 6,   5.473564, False),
    ('n-3f5/2', +3, 6, 6,   8.899377, False),
    ('n-4f5/2', +3, 7, 6,  12.839203, False),
    ('n-5f5/2', +3, 8, 6,  18.134527, False),
    # N.1g.7/2 ~ N.4g.7/2  (κ=+4, g 态 j=7/2, deg=8)
    ('n-1g7/2', +4, 5, 8,   3.463452, False),
    ('n-2g7/2', +4, 6, 8,   7.053631, False),
    ('n-3g7/2', +4, 7, 8,  11.503007, False),
    ('n-4g7/2', +4, 8, 8,  16.657375, False),

    # ═══════════════ 质子 (it=2) ═══════════════
    # P.1s.1/2 ~ P.6s.1/2  (κ=-1, s 态 j=1/2, deg=2)
    ('p-1s1/2', -1, 1, 2, -33.484013, True),
    ('p-2s1/2', -1, 2, 2,  -0.456185, True),
    ('p-3s1/2', -1, 3, 2,   1.979627, True),
    ('p-4s1/2', -1, 4, 2,   4.837749, True),
    ('p-5s1/2', -1, 5, 2,   9.022667, True),
    ('p-6s1/2', -1, 6, 2,  14.427224, True),
    # P.1p.3/2 ~ P.6p.3/2  (κ=-2, p 态 j=3/2, deg=4)
    ('p-1p3/2', -2, 2, 4, -16.971011, True),
    ('p-2p3/2', -2, 3, 4,   2.044221, True),
    ('p-3p3/2', -2, 4, 4,   4.286086, True),
    ('p-4p3/2', -2, 5, 4,   7.467313, True),
    ('p-5p3/2', -2, 6, 4,  11.862306, True),
    ('p-6p3/2', -2, 7, 4,  17.542119, True),
    # P.1d.5/2 ~ P.5d.5/2  (κ=-3, d 态 j=5/2, deg=6)
    ('p-1d5/2', -3, 3, 6,  -2.068293, True),
    ('p-2d5/2', -3, 4, 6,   2.662809, True),
    ('p-3d5/2', -3, 5, 6,   5.604965, True),
    ('p-4d5/2', -3, 6, 6,   9.672732, True),
    ('p-5d5/2', -3, 7, 6,  14.854870, True),
    # P.1f.7/2 ~ P.5f.7/2  (κ=-4, f 态 j=7/2, deg=8)
    ('p-1f7/2', -4, 4, 8,   3.383420, True),
    ('p-2f7/2', -4, 5, 8,   6.482551, True),
    ('p-3f7/2', -4, 6, 8,   9.287472, True),
    ('p-4f7/2', -4, 7, 8,  12.605528, True),
    ('p-5f7/2', -4, 8, 8,  17.988216, True),
    # P.1p.1/2 ~ P.6p.1/2  (κ=+1, p 态 j=1/2, deg=2)
    ('p-1p1/2', +1, 2, 2, -10.997589, True),
    ('p-2p1/2', +1, 3, 2,   2.067220, True),
    ('p-3p1/2', +1, 4, 2,   4.461084, True),
    ('p-4p1/2', +1, 5, 2,   7.879804, True),
    ('p-5p1/2', +1, 6, 2,  12.417871, True),
    ('p-6p1/2', +1, 7, 2,  18.132285, True),
    # P.1d.3/2 ~ P.5d.3/2  (κ=+2, d 态 j=3/2, deg=4)
    ('p-1d3/2', +2, 3, 4,   2.532002, True),
    ('p-2d3/2', +2, 4, 4,   3.607376, True),
    ('p-3d3/2', +2, 5, 4,   6.118755, True),
    ('p-4d3/2', +2, 6, 4,  10.356829, True),
    ('p-5d3/2', +2, 7, 4,  15.769058, True),
    # P.1f.5/2 ~ P.5f.5/2  (κ=+3, f 态 j=5/2, deg=6)
    ('p-1f5/2', +3, 4, 6,   3.386679, True),
    ('p-2f5/2', +3, 5, 6,   6.589596, True),
    ('p-3f5/2', +3, 6, 6,  10.353347, True),
    ('p-4f5/2', +3, 7, 6,  14.472924, True),
    ('p-5f5/2', +3, 8, 6,  19.673757, True),
    # P.1g.7/2 ~ P.4g.7/2  (κ=+4, g 态 j=7/2, deg=8)
    ('p-1g7/2', +4, 5, 8,   4.287728, True),
    ('p-2g7/2', +4, 6, 8,   8.057088, True),
    ('p-3g7/2', +4, 7, 8,  12.667373, True),
    ('p-4g7/2', +4, 8, 8,  17.996994, True),
]

# 仅占据态（E < 0 的束缚态，用于 SCF 密度计算）
OCCUPIED_STATES_16O = [s for s in ALL_STATES_16O if s[4] < 0]

# 通用态配置: 核素 → [(name, kappa, n_pr, degeneracy, E_init, is_proton), ...]
NUCLEUS_STATES = {
    '16O': ALL_STATES_16O,
}

# RMF 耦合常数 (保留兼容)
RMF_COUPLINGS = {
    'g_sigma': PKA1_PARAMS['gsig'],
    'g_omega': PKA1_PARAMS['gome'],
    'g_rho':   PKA1_PARAMS['grho'],
    'm_sigma': PKA1_PARAMS['amsig'],
    'm_omega': PKA1_PARAMS['amome'],
    'm_rho':   PKA1_PARAMS['amrho'],
}

# ═══════════════════════════════════════════════════════════════
#   PINN 网络超参数
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    'n_hidden': 128,         # 隐藏层维度
    'n_layers': 6,           # MLP 层数（不含输入输出）
    'activation': 'swish',   # 激活函数: 'swish' | 'tanh' | 'sin'
    'use_r_features': True,  # 启用 r 特征工程 (r, log(r), exp(-ar))
    'hard_normalize': True,  # 启用硬归一化约束 ∫(G²+F²)dr = 1
}

# 能量初始化策略
ENERGY_INIT_STRATEGY = 'fixed_value'  # 能量固定初始化 (势场从POT文件读取)

# ═══════════════════════════════════════════════════════════════
#   训练超参数
# ═══════════════════════════════════════════════════════════════

TRAIN_CONFIG = {
    # 优化器
    'lr': 1e-3,              # 学习率
    'weight_decay': 0,       # 权重衰减（PINN通常不用）
    
    # 训练轮次
    'max_epochs': 2000,      # 最大训练轮次（每轮=一次完整前向+反向）
    'early_stop_patience': 300,  # 早停耐心值
    
    # Batch: 对PINN通常全batch（所有网格点同时训练）
    'batch_size': None,      # None = 全 batch
    
    # 学习率调度
    'lr_scheduler': 'cosine', # 'cosine' | 'step' | 'none'
    'lr_decay_steps': 500,
    'lr_gamma': 0.5,
    
    # Adam betas (对PINN很重要)
    'adam_betas': (0.9, 0.999),
    
    # 梯度裁剪
    'grad_clip': 1.0,        # 梯度范数裁剪阈值, 0 表示不裁剪
}

# ═══════════════════════════════════════════════════════════════
#   损失函数权重
# ═══════════════════════════════════════════════════════════════

LOSS_WEIGHTS = {
    'pde': 1.0,              # Dirac方程PDE残差
    'norm': 10.0,            # 归一化约束 ∫(G²+F²)dr = 1
    'boundary_R': 1.0,       # r=R 截断处 G(R)=0 (Eq.3.60)
    'kinetic_positive': 0.1, # 正动能约束 (F分量不能主导)
    'node_count': 0.1,       # 节点数约束
    'rayleigh': 0.0,         # Rayleigh商能量一致性
    'ortho': 10.0,           # 波函数正交性约束 (用于激发态求解)
    # F 分量在 PDE 中额外加权
    'f_weight': 3.0,         # F残差相对于G残差的权重
}

# ═══════════════════════════════════════════════════════════════
#   SCF 外循环参数
# ═══════════════════════════════════════════════════════════════

SCF_CONFIG = {
    'max_iterations': 50,    # 最大 SCF 迭代次数
    'convergence_eps': 1e-5, # 收敛判据: 总能量变化 < eps (MeV)
    'xmix_initial': 0.3,     # 初始混合系数
    'xmix_min': 0.05,        # 最小混合系数
    'xmix_max': 1.0,         # 最大混合系数
    'xmix_growth': 1.0618,   # 混合系数增长率 (黄金比例相关)
    
    # 每个 SCF 步内 PINN 训练
    'pinn_epochs_per_scf': 1000,  # 每次 SCF 迭代的 PINN 训练轮次
    'pinn_lr_per_scf': 5e-4,      # SCF 迭代中的 PINN 学习率
    
    # 收敛监控
    'monitor_quantities': ['total_energy', 'rms_radius', 'potential_diff'],
}

# ═══════════════════════════════════════════════════════════════
#   设备和随机种子
# ═══════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def get_device():
    """返回当前可用设备"""
    return DEVICE


def set_seed(seed=SEED):
    """设置全局随机种子以确保可复现性"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_isotope_config(isotope_name=None):
    """获取核素配置，默认使用 DEFAULT_ISOTOPE"""
    if isotope_name is None:
        isotope_name = DEFAULT_ISOTOPE
    if isotope_name in ISOTOPE_CONFIG:
        return ISOTOPE_CONFIG[isotope_name]
    raise ValueError(f"未知核素: {isotope_name}, 可用: {list(ISOTOPE_CONFIG.keys())}")


def get_r_grid(device=None):
    """返回径向网格张量"""
    r = torch.tensor(R_GRID, dtype=torch.float32)
    if device is not None:
        r = r.to(device)
    return r


def get_r_safe(device=None):
    """返回安全的径向网格张量(r=0处有小偏移)"""
    r = torch.tensor(R_SAFE, dtype=torch.float32)
    if device is not None:
        r = r.to(device)
    return r
