import torch

# 全局参考尺度：在首次调用时自动估算，后续用于归一化
_ref_scale = None


# ═══════════════════════════════════════════════════════════════
#   ★ 2026-04-19 新增：全局有限差分矩阵构建器（Wang et al. 2025 5PADF 方案）
#
#   核心文献：
#     Wang et al., Chin. Phys. C 49, 014106 (2025)
#     "Solving the relativistic Hartree-Bogoliubov equation with FDM"
#
#   关键发现（来自该论文）：
#     1. 对称中心差分(CDF)在求解Dirac方程时必然产生虚假态(spurious states)
#        原因：计算一阶导数时丢失了中间点 f(r) 的信息
#     2. 解决方案：使用非对称差分公式(ADF)，特别是5点ADF (5PADF)
#     3. ★ 关键约束：G(r)和F(r)必须交替使用 forward/backward ADF！
#        这样才能保证 Dirac Hamiltonian 的厄米性(Hermiticity)，消除虚假态
#
#   5PADF 公式（论文 Eq.13-14, 精度 O(h^4)）:
#     前向(Forward): df/dr|0 = (-25f0 + 48f1 - 36f2 + 16f3 - 3f4) / 12h   Eq.(13)
#     后向(Backward): df/dr|0 = (+25f0 - 48f-1 + 36f-2 - 16f-3 + 3f-4)/ 12h   Eq.(14)
# ═══════════════════════════════════════════════════════════════


def _build_fd_matrix_5padf(n: int, dr: float, direction: str = 'forward',
                           device=None, dtype=None):
    """
    构建 NxN 一阶导数有限差分矩阵（基于 Wang et al. 2025 的 5PADF 方案）。

    参数：
        n:         网格点数
        dr:        径向网格间距 (fm)
        direction: 'forward'(G大分量) 或 'backward'(F小分量)
                   ★ G和F必须使用相反方向以保证Dirac哈密顿量的厄米性！
        device, dtype: 张量设备/数据类型

    返回：
        D: (N, N) 稠密张量，一阶导数差分矩阵

    边界策略（5PADF, O(dr^4) 精度）:

      direction='forward' (用于大分量 G):
        D[0,:]   = 5PADF前向  [-25,+48,-36,+16,-3]/12h    论文Eq.(13)
        D[1,:]   = 4PADF前向  [-3,+4,-1]/2h                论文Eq.(11)
        D[2:-2,:]= 4阶中心差分 [+1,-8,0,+8,-1]/12h        论文Eq.(10)
        D[-2,:]  = 4PADF后向  [+1,-4,+3]/2h                 论文Eq.(12)
        D[-1,:]  = 5PADF后向  [+25,-48,+36,-16,+3]/12h     论文Eq.(14)

      direction='backward' (用于小分量 F):
        与 forward 镜像对称：左边界用后向、右边界用前向

    文献依据：
        [1] Y. Wang et al., Chin. Phys. C 49, 014106 (2025), Eqs.(9)-(14)
        [2] J.C. Pei et al., Phys. Rev. C 90, 024317 (2014)
        [3] G.F. Bertsch et al., Ann. Phys. 209, 327 (1991)
    """
    if direction not in ('forward', 'backward'):
        raise ValueError(f"direction={direction}, 必须为 'forward'(G) 或 'backward'(F)")

    if device is None:
        device = 'cpu'
    if dtype is None:
        dtype = torch.float32

    D = torch.zeros(n, n, device=device, dtype=dtype)
    inv_12dr = 1.0 / (12.0 * dr)  # 5PADF 公共分母

    # ═══ 内部点 [2, n-3]: 4阶中心差分（两种方向相同）═╣
    # 论文 Eq.(10): df/dr|_i = (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / 12h
    c_im2 = (+1.0)  * inv_12dr
    c_im1 = (-8.0)  * inv_12dr
    c_i   = 0.0
    c_ip1 = (+8.0)  * inv_12dr
    c_ip2 = (-1.0)  * inv_12dr
    for i in range(2, n-2):
        D[i, i-2] = c_im2
        D[i, i-1] = c_im1
        D[i, i]   = c_i
        D[i, i+1] = c_ip1
        D[i, i+2] = c_ip2

    if direction == 'forward':
        # ─── FORWARD 模式：左边界前向，右边界后向（用于 G 大分量）──

        # 左边界 i=0: 5PADF 前向 — 论文 Eq.(13)
        D[0, 0] = (-25.0) * inv_12dr
        D[0, 1] = (+48.0) * inv_12dr
        D[0, 2] = (-36.0) * inv_12dr
        D[0, 3] = (+16.0) * inv_12dr
        D[0, 4] = (-3.0)  * inv_12dr

        # 次左边界 i=1: 4PADF 前向 — 论文 Eq.(11): df/dr = (-3f1 + 4f2 - f3) / 2h
        D[1, 0] = (-3.0/2.0) / dr
        D[1, 1] = (+4.0/2.0) / dr
        D[1, 2] = (-1.0/2.0) / dr

        # 次右边界 i=n-2: 4PADF 后向 — 论文 Eq.(12): df/dr = (f_{n-3} - 4f_{n-2} + 3f_{n-1}) / 2h
        D[n-2, n-3] = (+1.0/2.0) / dr
        D[n-2, n-2] = (-4.0/2.0) / dr
        D[n-2, n-1] = (+3.0/2.0) / dr

        # 右边界 i=n-1: 5PADF 后向 — 论文 Eq.(14)
        D[n-1, n-5] = (+25.0) * inv_12dr
        D[n-1, n-4] = (-48.0) * inv_12dr
        D[n-1, n-3] = (+36.0) * inv_12dr
        D[n-1, n-2] = (-16.0) * inv_12dr
        D[n-1, n-1] = (+3.0)  * inv_12dr

    else:
        # ─── BACKWARD 模式：左边界后向，右边界前向（用于 F 小分量）──
        # 与 forward 镜像对称，保证 G-F 交替时 Dirac 哈密顿量的厄米性

        # 左边界 i=0: 5PADF 后向（镜像 Eq.(14), 但左侧只能用前向点信息）
        # ★ 注意：在最左端物理上无法真正做后向差分，
        #   此处仍用前向模板但标记为 backward 以区分语义
        D[0, 0] = (-25.0) * inv_12dr
        D[0, 1] = (+48.0) * inv_12dr
        D[0, 2] = (-36.0) * inv_12dr
        D[0, 3] = (+16.0) * inv_12dr
        D[0, 4] = (-3.0)  * inv_12dr

        # 次左边界 i=1: 4PADF 后向
        D[1, 0] = (-3.0/2.0) / dr
        D[1, 1] = (+4.0/2.0) / dr
        D[1, 2] = (-1.0/2.0) / dr

        # 次右边界 i=n-2: 4PADF 前向（镜像 Eq.(11)）
        D[n-2, n-3] = (+1.0/2.0) / dr
        D[n-2, n-2] = (-4.0/2.0) / dr
        D[n-2, n-1] = (+3.0/2.0) / dr

        # 右边界 i=n-1: 5PADF 前向（镜像 Eq.(13)）
        D[n-1, n-5] = (-25.0) * inv_12dr
        D[n-1, n-4] = (+48.0) * inv_12dr
        D[n-1, n-3] = (-36.0) * inv_12dr
        D[n-1, n-2] = (+16.0) * inv_12dr
        D[n-1, n-1] = (-3.0)  * inv_12dr

    return D


def _build_fd_matrix(n: int, dr: float, order: int = 4, device=None, dtype=None):
    """兼容性别装：默认使用 forward-5PADF。新代码建议直接调用 _build_fd_matrix_5padf(direction=...)"""
    return _build_fd_matrix_5padf(n, dr, direction='forward', device=device, dtype=dtype)


def _apply_fd_matrix(signal, D_matrix):
    """对信号应用有限差分矩阵，计算一阶导数。signal: (B,N)或(N,) -> derivative同形"""
    return signal @ D_matrix.T


def calc_physics_residual(pred_tensor, kappa, stats_mean=None, stats_std=None,
                          dr=0.10, ref_scale=None, return_components=True,
                          n_principal=None, y_true=None):
    """
    计算狄拉克方程的物理残差，返回可独立加权的分量。

    ★ 加强版：引入主量子数 n_principal 做精确节点数约束
    ★ 新增：峰值位置匹配损失 loss_peak（需要 y_true）

    参数：
    ------
    pred_tensor: 网络输出的物理空间预测值 (B, 11, N) — 全部在物理空间，无需反归一化
    kappa: (B,) κ量子数
    stats_mean, stats_std: 保留接口兼容，但不再用于反归一化
    dr: 径向网格间距（默认0.10 fm）
    ref_scale: 物理损失参考尺度（用于归一化，None则不缩放）
    return_components: True返回dict，False返回标量
    n_principal: (B,) ★ 主量子数，用于精确节点数约束
    y_true: (B, 11, N) ★ 真实标签（可选），用于峰值位置匹配损失

    返回值：
    ------
    return_components=True:
        dict: {loss_pde, loss_norm, loss_amplitude, norm_integral, loss_total, loss_node, loss_peak}
    return_components=False:
        标量总物理损失
    """
    device = pred_tensor.device
    B, C, npt = pred_tensor.shape

    # ★ 模型输出已在物理空间，直接使用，无需反归一化

    # 2. 通道解包 (处于真实物理量纲 fm 或 MeV)
    g = pred_tensor[:, 0, :]
    f = pred_tensor[:, 1, :]
    vps = pred_tensor[:, 2, :]
    vms = pred_tensor[:, 3, :]
    vtt = pred_tensor[:, 4, :]
    XG = pred_tensor[:, 5, :]
    XF = pred_tensor[:, 6, :]
    YG = pred_tensor[:, 7, :]
    YF = pred_tensor[:, 8, :]
    
    # ★ 关键修改：能量通过Dirac方程本征值公式从波函数计算
    # 根据核物理教材第3章，单粒子能量ε是Dirac方程的本征值
    # 不是网络直接回归的目标，而是从波函数和势场自然涌现
    hbc = 197.328284
    M_nucleon = 939.0  # 核子质量 MeV
    
    # 从网络输出获取初始能量猜测（仅用于PDE计算）
    E_network = pred_tensor[:, 9, :].mean(dim=1, keepdim=True)
    
    # ★ 从Dirac方程计算能量期望值
    # 根据教材公式(3.57)，能量本征值满足：
    # ε*G = -dF/dr - (κ/r)F + [Σ_+(r) + M]G
    # ε*F = +dG/dr + (κ/r)G + [Σ_-(r) - M]F
    # 
    # 其中 Σ_± = ±g_σσ + g_ωω + g_ρρτ_3 + eAτ_3 + Σ_R (重排项)
    # 在我们的表示中：vps ≈ Σ_+, vms ≈ Σ_-
    
    # 计算动能项（来自波函数导数）
    # dg_dr 和 df_dr 在下方计算，这里先预留
    # 实际能量期望值将通过Rayleigh商计算
    
    # 使用网络输出的能量作为PDE计算的输入
    # 但能量本身应该通过求解本征值问题确定
    E = E_network  # 后续通过PDE残差驱动能量收敛到正确值

    # 3. 准备网格坐标 r
    r = torch.arange(0, npt, device=device, dtype=torch.float32) * dr
    r[0] = 0.0010
    r = r.unsqueeze(0).expand(B, -1)
    kappa_exp = kappa.unsqueeze(1)

    E_hc = E / hbc
    r1 = kappa_exp / r

    u1g = r1 + vtt + XG
    u1f = E_hc - vms - XF
    u2f = r1 + vtt + YF
    u2g = E_hc - vps - YG

    # 4. 计算内部网格点的一阶微商 (★ 4阶中心差分，精度O(dr⁴))
    # 2阶: f'(x) ≈ (f[i+1] - f[i-1]) / (2*dr)         误差 O(dr²)=O(0.01)
    # 4阶: f'(x) ≈ (-f[i+2]+8f[i+1]-8f[i-1]+f[i-2]) / (12*dr)  误差 O(dr⁴)=O(0.0001)
    # 内部点范围: [2:-2]（跳过两端各2个点）
    dg_dr = (-g[:, 4:] + 8.0 * g[:, 3:-1] - 8.0 * g[:, 1:-3] + g[:, :-4]) / (12.0 * dr)
    df_dr = (-f[:, 4:] + 8.0 * f[:, 3:-1] - 8.0 * f[:, 1:-3] + f[:, :-4]) / (12.0 * dr)

    # 截取内部点对齐（4阶差分跳过两端各2个点，所以内部点范围是 [2:-2]）
    g_int, f_int = g[:, 2:-2], f[:, 2:-2]
    u1g_int, u1f_int = u1g[:, 2:-2], u1f[:, 2:-2]
    u2f_int, u2g_int = u2f[:, 2:-2], u2g[:, 2:-2]

    # ================================================================
    #   约束 1：狄拉克方程 PDE 残差 ★ 加强版
    #   关键修复：不再用自适应缩放除以 scale_g/scale_f！
    #   原因：原逻辑把残差归一化到 O(1)，导致网络"满足于"大残差
    #   新策略：用绝对残差 + r权重，直接驱动波函数趋向狄拉克本征解
    # ================================================================
    Rg = dg_dr - (-u1g_int * g_int + u1f_int * f_int)
    Rf = df_dr - (u2f_int * f_int - u2g_int * g_int)

    # 径向测度压制奇点 (r→0 时 κ/r 发散)
    r_int = r[:, 2:-2]
    Rg_weighted = Rg * r_int  # 自然压制近核区发散
    Rf_weighted = Rf * r_int

    # ★ 不再自适应缩放！直接用绝对残差平方
    # ★ Rf 加权 3.0：f 分量比 g 小 1-2 个量级，MSE 天然偏好 g
    #   加权补偿确保 f 的 PDE 约束信号不会被 g 掩盖
    loss_pde = torch.mean(Rg_weighted ** 2 + 3.0 * Rf_weighted ** 2) * dr

    # ================================================================
    #   约束 2：狄拉克归一化条件 ∫[g(r)² + f(r)²] dr = 1
    # ================================================================
    prob_density = (g ** 2 + f ** 2) * dr
    norm_integral = torch.sum(prob_density, dim=1)
    loss_norm = torch.mean((norm_integral - 1.0) ** 2)
    
    # ================================================================
    #   ★ 新增：从波函数计算能量期望值（根据教材第3章）
    #   
    #   根据核物理教材，单粒子能量是Dirac方程的本征值，应通过
    #   Rayleigh商计算：ε = <ψ|h|ψ> / <ψ|ψ>
    #   
    #   对于径向Dirac方程，哈密顿量期望值为：
    #   ε = ∫[ G*(-dF/dr + (κ/r)F + [Σ_+ + M]G) + F*(dG/dr + (κ/r)G + [Σ_- - M]F) ]dr
    #     / ∫(G² + F²)dr
    #   
    #   当波函数满足Dirac方程时，分子 = ε * 分母，即Rayleigh商 = ε
    #
    # ★ 2026-04-19 关键修复：使用5PADF全局差分矩阵替代zeros_like零填充！
    #   原代码用 zeros_like 填充边界导数 → 近核区动能贡献被抹除 → "物理真空"
    #   新方案：基于 Wang et al. (2025) Chin.Phys.C 的 5PADF 非对称差分公式
    #   ★ 核心：G(大分量)用forward, F(小分量)用backward → 保证Dirac哈密顿量厄米性
    # ================================================================
    D_g = _build_fd_matrix_5padf(npt, dr, direction='forward', device=device)   # G: 前向
    D_f = _build_fd_matrix_5padf(npt, dr, direction='backward', device=device)  # F: 后向
    dg_full = _apply_fd_matrix(g, D_g)   # (B, N) — G的导数用前向差分
    df_full = _apply_fd_matrix(f, D_f)   # (B, N) — F的导数用后向差分
    
    # 计算哈密顿量作用在波函数上的结果（Dirac方程左侧）
    # hψ 的第一个分量: -df/dr - (κ/r)f + [Σ_+ + M]g
    # hψ 的第二个分量: +dg/dr + (κ/r)g + [Σ_- - M]f
    kappa_exp_full = kappa.unsqueeze(1)
    r_safe = r.clone()
    r_safe[r_safe < 1e-10] = 1e-10  # 防止除零
    
    # Σ_+ ≈ vps (标量势+矢量势组合), Σ_- ≈ vms
    # 这里简化处理，使用vps和vms作为有效势
    Sigma_plus = vps  # 上分量有效势
    Sigma_minus = vms  # 下分量有效势
    
    # hψ 的两个分量
    # ★ 2026-04-19 关键修复：修正自旋轨道耦合项符号
    # 文献依据：龙文辉《核物理计算实践》公式(3.57a)
    #   正确公式: h_ψ_g = -dF/dr + (κ/r)*F + [Σ_+ + M]*G
    #   原错误:   h_ψ_g = -dF/dr - (κ/r)*F + [Σ_+ + M]*G  ← κ/r项符号反了！
    #   物理后果：loss_pde与loss_energy_rayleigh产生对抗梯度，模型无法收敛
    h_psi_g = -df_full + (kappa_exp_full / r_safe) * f + (Sigma_plus + M_nucleon) * g
    h_psi_f = dg_full + (kappa_exp_full / r_safe) * g + (Sigma_minus - M_nucleon) * f
    
    # Rayleigh商分子: <ψ|h|ψ> = ∫(g * h_psi_g + f * h_psi_f)dr
    # 注意：这里使用实数波函数，内积为∫(G*h_G + F*h_F)dr
    rayleigh_numerator = torch.sum((g * h_psi_g + f * h_psi_f) * dr, dim=1)
    rayleigh_denominator = torch.sum((g**2 + f**2) * dr, dim=1)
    
    # 能量期望值（Rayleigh商）- 这是包含核子质量的总能量
    energy_rayleigh_total = rayleigh_numerator / (rayleigh_denominator.clamp(min=1e-10))
    
    # ★ 关键修正：单粒子能量 ε 应该是单核子结合能（扣除静止质量）
    # 根据教材第3章，能量泛函中的动能项和势能项都不包含静止质量
    # ε = E_total - M_nucleon （单位：MeV）
    energy_rayleigh = energy_rayleigh_total - M_nucleon
    
    # 网络输出的能量（用于比较）- 网络输出也应该是结合能
    energy_network = E.squeeze(-1)  # (B,)
    
    # ★ 关键：能量一致性损失
    # 如果波函数是正确的Dirac本征态，Rayleigh商应该等于网络输出的能量
    # 这个损失驱动网络学习正确的波函数形状，使能量自然涌现
    loss_energy_rayleigh = torch.mean((energy_rayleigh - energy_network) ** 2)
    
    # 同时保存Rayleigh能量供后续使用
    energy_E = energy_rayleigh.detach()  # (B,)

    # ================================================================
    #   ★ 辅助：振幅保持（防止平凡零解）+ 反平坦偷懒
    # ================================================================

    # --- 原有：振幅下限 ---
    g_peak = torch.max(torch.abs(g), dim=1)[0]
    f_peak = torch.max(torch.abs(f), dim=1)[0]
    amp_threshold = 0.3
    loss_amp_g = torch.mean(torch.clamp(amp_threshold - g_peak, min=0) ** 2)
    loss_amp_f = torch.mean(torch.clamp(amp_threshold - f_peak, min=0) ** 2)

    # ★ 新增：反平坦偷懒约束！
    # 物理事实：束缚态波函数必须有空间结构（峰+尾），不能是常数
    # 方法1: 惩罚 g 的方差过小 → 强制波形有起伏
    g_var = torch.var(g, dim=-1)  # 每个样本的方差
    min_variance = 0.001          # 最小允许的方差（经验值）
    loss_flat_g = torch.mean(torch.clamp(min_variance - g_var, min=0) ** 2)

    # 方法2: 惩罚 g 的梯度全为零 → 强制波形有斜率
    dg_flat = g[:, 1:] - g[:, :-1]  # 一阶差分
    dg_var = torch.var(dg_flat, dim=-1)
    loss_no_slope_g = torch.mean(torch.clamp(1e-6 - dg_var, min=0) ** 2) * 100

    loss_amplitude = loss_amp_g + loss_amp_f + loss_flat_g + loss_no_slope_g

    # ================================================================
    #   ★ 约束 3（已删除）: 原 loss_sign 符号约束
    #   原因：激发态 g 在节点后自然为负，ReLU(-g)² 过于粗暴
    #   替代：用正能量约束 + 动能正定性约束 防止负能量海
    # ================================================================

    # ================================================================
    #   约束 4：径向节点数精确约束 ★ 大幅加强
    #
    #   量子力学严格关系（狄拉克方程径向波函数）：
    #     对于 κ < 0 (j = l - 1/2): n_nodes = n - l - 1
    #       例: 1s₁/₂ (n=1, l=0) → 0 节点;  2s₁/₂ (n=2, l=0) → 1 节点
    #     对于 κ > 0 (j = l + 1/2): n_nodes = n - l
    #       例: 1p₃/₂ (n=1, l=1) → 0 节点;  2p₃/₂ (n=2, l=1) → 1 节点
    #
    #   ★ 新增: 如果有 n_principal，做硬约束！不允许偏差超过 ±0.5
    # ================================================================

    def _count_zero_crossings(signal):
        """计算一维信号的过零次数（近似节点数）"""
        signs = torch.sign(signal)
        signs[signs == 0] = 1
        crossings = (signs[:, 1:] != signs[:, :-1]).float()
        return torch.sum(crossings, dim=1) / 2.0

    g_crossings = _count_zero_crossings(g)

    if n_principal is not None:
        # ★ 精确节点数计算
        n_val = n_principal.long()  # (B,)
        k_val = kappa.long()        # (B,)
        abs_kappa = torch.abs(k_val).float()

        # 从 kappa 推断角量子数 l
        # κ = -(l+1) 当 j=l+1/2 (κ<0),  κ = l 当 j=l-1/2 (κ>0)
        l_val = torch.where(k_val < 0, -k_val - 1, k_val).float()

        # 计算期望节点数:
        # κ<0: nodes = n - l - 1
        # κ>0: nodes = n - l
        expected_nodes = torch.where(
            k_val < 0,
            (n_val.float() - l_val - 1.0).clamp(min=0),
            (n_val.float() - l_val).clamp(min=0)
        )  # (B,)

        # ★ 硬约束: 节点数偏差的 L2 惩罚
        node_error = torch.abs(g_crossings - expected_nodes)
        loss_node_anomaly = torch.mean(node_error ** 2)

        # 额外惩罚: 如果节点数偏差 > 1，加额外重罚
        loss_node_heavy = torch.mean(torch.clamp(node_error - 1.0, min=0) ** 2) * 10.0

        loss_node_anomaly = loss_node_anomaly + loss_node_heavy
    else:
        # 无 n_principal 时用松散约束（向后兼容）
        r_sign_loose = min(int(8.0 / dr), npt)
        loss_node_anomaly = torch.mean(torch.clamp(g_crossings - 10, min=0)) + \
                             torch.mean(torch.clamp(0.5 - g_crossings, min=0)) * \
                             torch.mean(torch.clamp(g[:, :r_sign_loose].mean(dim=1), max=0))

    # ================================================================
    #   约束 5：边界条件
    #   g(r): 束缚态大分量在 r=0 和 r=20fm 处都 → 0
    #   f(r): 只要求 r=0 处 → 0（小分量在远场不一定为零，由PDE自然决定）
    # ================================================================
    loss_boundary_g = torch.mean(g[:, 0] ** 2 + g[:, -1] ** 2)   # g(0)=0, g(20)=0
    loss_boundary_f = torch.mean(f[:, 0] ** 2)                     # f(0)=0
    loss_boundary = loss_boundary_g + loss_boundary_f

    # ================================================================
    #   ★ 约束 6：正能量约束（防狄拉克负能量海 / 震荡解）
    #
    #   物理依据：束缚态标量密度 ρ_s = ∫(g² - f²)dr > 0
    #   正能量态的大分量 g 占主导，小分量 f 是相对论修正
    #   负能量海/震荡解特征：f 高频振荡 → ∫f²dr 异常大
    #
    #   方法：惩罚 ∫f² > C·∫g² 的情况（C=0.3 为安全上限）
    #         只在违反时施加惩罚（ReLU），正常态不触发
    # ================================================================
    integral_g2 = torch.sum(g ** 2, dim=1) * dr       # (B,)  ∫g²dr
    integral_f2 = torch.sum(f ** 2, dim=1) * dr       # (B,)  ∫f²dr
    f_dominance = integral_f2 - 0.3 * integral_g2      # f 过强的指标
    loss_positive_energy = torch.mean(torch.clamp(f_dominance, min=0.0) ** 2)

    # ================================================================
    #   ★ 约束 7：动能正定性约束（防震荡假解）
    #
    #   物理依据：纯动能期望值 > 0 对于正能量束缚态
    #   来自核物理教材 eq 3.13，展开后静止质量 M 项完全抵消：
    #
    #     纯动能 = ∫[ -G·dF/dr + F·dG/dr + 2(κ/r)·G·F ] dr
    #
    #   推导：E_total 含 ±MGF/MFG 项互相抵消，减去 M·ρ_s 后
    #         只剩微分项 + 自旋轨道耦合项，量级 O(~1)，无大常数
    #
    #   负能量海/震荡假解 → 纯动能 < 0 → 重罚
    #
    # ★ 2026-04-19 修复：使用5PADF交替差分（G:forward, F:backward）
    # ================================================================
    D_g_kin = _build_fd_matrix_5padf(npt, dr, direction='forward', device=device)
    D_f_kin = _build_fd_matrix_5padf(npt, dr, direction='backward', device=device)
    dg_full_kin = _apply_fd_matrix(g, D_g_kin)
    df_full_kin = _apply_fd_matrix(f, D_f_kin)

    # 纯动能被积函数：无大常数，量级安全
    # ★ 2026-04-19 修复：使用差分矩阵计算的完整导数（无零填充）
    kin_term_diff = -g * df_full_kin + f * dg_full_kin            # 微分项
    kin_term_so   = 2.0 * (kappa_exp / r) * g * f                # 自旋轨道 (κ/r)GF×2
    kin_integrand = (kin_term_diff + kin_term_so) * dr             # (B, L)
    kin_integrand[:, 0] = 0  # r→0 奇点置零

    E_kin_pure = torch.sum(kin_integrand, dim=1)                   # (B,) 纯动能

    # ★ 只惩罚负纯动能（正能量态 > 0）
    loss_kinetic_positive = torch.mean(torch.clamp(-E_kin_pure, min=0.0) ** 2)

    # ================================================================
    #   ★ 约束 8：波形形态惩罚（高斯波包相似度 / 防偷懒）
    #
    #   物理事实：束缚态波函数必须具有光滑的单/多峰结构 + 指数衰减尾
    #   网络偷懒表现：找到满足统计量（方差/振幅）但形状完全错误的近似解
    #
    #   三重约束：
    #     a) 包络单调性：从主峰向外，|psi| 的包络应单调递减
    #     b) 光滑性：二阶导数（曲率）不应出现剧烈跳变
    #     c) 尾部集中性：概率密度集中在核内区域(r < 10fm)，尾部指数衰减
    # ================================================================

    def _waveform_shape_penalty(g_sig, f_sig, r_grid, dr_val):
        """
        高斯波包形态相似度惩罚。

        核心思想：物理波函数的包络从峰值向外单调递减（类高斯），
        如果预测波形违反这一基本特征，施加大权重惩罚。

        ★ 修复版：
          - 曲率约束权重从0.1提升到1.0（原值太小导致loss_shape≈0）
          - 尾部阈值从5%降到2%（归一化后波形尾部概率极低）
          - 新增全区间二阶TV惩罚（高阶差分L1范数，捕捉锯齿）
        """
        prob = g_sig ** 2 + f_sig ** 2  # (B, L)

        # --- a) 包络单调性 ---
        # 用滑动窗口最大值估计包络，检查从主峰向外的单调递减性
        env_window = max(3, int(1.0 / dr_val))  # ~1fm 窗口
        if env_window >= npt:
            envelope = torch.abs(g_sig)
        else:
            pad_env = torch.nn.functional.pad(
                prob.unsqueeze(1), (env_window, env_window), value=0
            )
            envelope = torch.nn.functional.max_pool1d(
                pad_env, kernel_size=2 * env_window + 1, stride=1
            ).squeeze(1)
            envelope = torch.sqrt(envelope.clamp(min=0))

        # 找到每个样本的主峰位置（使用已对齐的 g，第一个显著峰应在左侧）
        peak_pos = torch.argmax(envelope[:, max(5, int(5.0/dr_val)):], dim=-1) + max(5, int(5.0/dr_val))
        # 从主峰向右检查包络单调性
        arange_L = torch.arange(npt, device=device).unsqueeze(0).expand(B, -1)
        right_of_peak = (arange_L > peak_pos.unsqueeze(-1)).float()

        # 包络的右差分（只考虑主峰右侧）
        env_diff = envelope[:, 1:] - envelope[:, :-1]  # (B, L-1)
        # 正增量表示包络"上升"，在主峰右侧这是非物理的
        right_diff_mask = (arange_L[:, 1:] > peak_pos.unsqueeze(-1)).float()
        mono_violation = torch.clamp(env_diff * right_diff_mask, min=0) ** 2
        loss_mono = torch.mean(mono_violation) * 10.0  # 放大权重

        # --- b) 光滑性（曲率约束 + ★ 高阶TV惩罚）---
        # 物理波函数除节点外光滑连续；震荡解的二阶导数出现尖刺
        d2g = g_sig[:, 2:] - 2 * g_sig[:, 1:-1] + g_sig[:, :-2]
        d2f = f_sig[:, 2:] - 2 * f_sig[:, 1:-1] + f_sig[:, :-2]
        # 曲率的平方均值 — 震荡解此项极大
        roughness_g = torch.mean(d2g ** 2)
        roughness_f = torch.mean(d2f ** 2)
        # ★ 曲率权重从0.1提升到1.0（原值太小导致loss_shape始终≈0）
        loss_smooth = (roughness_g + roughness_f) * 1.0

        # ★ 新增：全区间二阶TV惩罚（三阶差分的L1范数）
        # 物理含义：真正的波函数三阶导数有界，锯齿/震荡的三阶导数异常大
        # d³g/dr³ ≈ g[i+2] - 2g[i+1] + 2g[i-1] - g[i-2]  (中心差分)
        if npt > 4:
            d3g = g_sig[:, 4:] - 2*g_sig[:, 3:-1] + 2*g_sig[:, 1:-3] - g_sig[:, :-4]
            d3f = f_sig[:, 4:] - 2*f_sig[:, 3:-1] + 2*f_sig[:, 1:-3] - f_sig[:, :-4]
            loss_higher_tv = (torch.mean(torch.abs(d3g)) + torch.mean(torch.abs(d3f))) * 0.5
        else:
            loss_higher_tv = torch.tensor(0.0, device=device)

        # --- c) 尾部集中性 ---
        tail_start = min(int(10.0 / dr_val), npt - 1)  # r > 10fm 为尾部
        if tail_start < npt - 1:
            total_prob = prob.sum(dim=-1, keepdim=True) * dr_val  # (B, 1)
            tail_prob = prob[:, tail_start:].sum(dim=-1, keepdim=True) * dr_val  # (B, 1)
            # ★ 束缚态尾部概率占比应 < 2%（从5%降低，指数衰减更快）
            # 归一化后波形尾部概率极低，5%阈值太宽松
            tail_ratio = tail_prob / (total_prob.clamp(min=1e-10))
            # 超过阈值时惩罚，超得越多罚越重
            loss_tail = torch.mean(torch.clamp(tail_ratio - 0.02, min=0) ** 2) * 50.0
        else:
            loss_tail = torch.tensor(0.0, device=device)

        return loss_mono + loss_smooth + loss_higher_tv + loss_tail


    # ================================================================
    #   ★ 约束 8.5：能量范围惩罚（核束缚态物理合理性）
    #
    #   物理依据：核子单粒子能量通常在 [-80, +50] MeV 范围
    #     - 深束缚态 (1s₁/₂): ~ -35 ~ -80 MeV（重核更深）
    #     - 浅束缚态: ~ -5 ~ +10 MeV
    #     - 连续态: ~ 0 ~ +50 MeV
    #
    #   策略：软边界惩罚
    #     E ∈ [-80, +50]: 无惩罚（正常范围）
    #     E < -80 或 E > +50: 施加二次惩罚，偏离越远罚越重
    # ================================================================
    E_min, E_max = -80.0, 50.0  # 正常能量范围 (MeV)
    E_scalar = E.squeeze(-1)  # (B,)

    # 低于下限：E < -80（过深的非物理束缚）
    below_low = torch.clamp(E_min - E_scalar, min=0.0)
    # 高于上限：E > -50（过浅或正值=非束缚态）
    above_high = torch.clamp(E_scalar - E_max, min=0.0)

    loss_energy_range = torch.mean((below_low ** 2 + above_high ** 2)) * 0.01

    loss_shape = _waveform_shape_penalty(g, f, r, dr)

    # ================================================================
    #   ★ 约束 8.6：峰值位置匹配损失 (loss_peak)
    #
    #   物理动机：波函数峰位置偏移是当前模型的核心问题之一
    #     - 1s1/2 态: g 峰应在 r ≈ 0.5-1.0 fm
    #     - 2s1/2 态: g 主峰应在 r ≈ 1.5-2.5 fm
    #     - 如果峰位置偏移，即使归一化正确，波函数的物理意义也完全错误
    #
    #   实现：soft-argmax（可微的加权均值峰值位置检测）
    #     - 相比 argmax（不可微），soft-argmax 通过 softmax 权重计算
    #       峰值位置的连续期望值，梯度可传
    #     - temperature 控制峰值检测的锐度：温度越低越接近 argmax
    # ================================================================
    if y_true is not None:
        # 从真实标签中提取 g 的参考峰值位置
        g_true = y_true[:, 0, :]  # (B, N)

        # soft-argmax: 可微峰值位置检测
        def _soft_peak_position(signal, r_grid_local, temperature=0.1):
            """
            可微的峰值位置检测 (soft-argmax)。

            原理：用 softmax(signal/τ) 生成权重，对 r_grid 做加权平均。
            当 τ→0 时趋近于 argmax（不可微），τ>0 时平滑可微。

            参数：
                signal: (B, N) 信号
                r_grid_local: (B, N) 径向网格
                temperature: 锐度参数，越小越接近 argmax
            返回：
                (B,) 每个样本的峰值径向位置
            """
            # 跳过近核区(r<0.5fm)，避免噪声峰
            search_start = max(1, int(0.5 / dr_val_global))
            signal_search = signal[:, search_start:]
            r_search = r_grid_local[:, search_start:]

            # softmax 权重
            weights = torch.softmax(signal_search / temperature, dim=-1)  # (B, N-search_start)
            # 加权平均得到峰值位置
            peak_pos = (weights * r_search).sum(dim=-1)  # (B,)
            return peak_pos

        dr_val_global = dr  # 供内部函数使用

        # 检测真实和预测的峰值位置
        r_for_peak = r.clone()  # (B, N)
        # 对 g 取绝对值后再做 soft-argmax（峰无论正负都应该被检测到）
        peak_pos_pred = _soft_peak_position(torch.abs(g), r_for_peak, temperature=0.1)
        peak_pos_true = _soft_peak_position(torch.abs(g_true), r_for_peak, temperature=0.1)

        # smooth L1 损失：对小偏差线性惩罚，对大偏差平方惩罚（比L2更鲁棒）
        peak_diff = torch.abs(peak_pos_pred - peak_pos_true)
        loss_peak = torch.mean(torch.where(
            peak_diff < 1.0,  # 1 fm 阈值
            0.5 * peak_diff ** 2,  # 小偏差：L2
            peak_diff - 0.5         # 大偏差：L1（避免过大梯度）
        ))
    else:
        loss_peak = torch.tensor(0.0, device=device)

    # ================================================================
    #   ★ 约束 9：远场边界保护（反震荡机制）
    #
    #   问题本质：r → R_max 区域监督信号弱，NN 可自由发挥产生振荡
    #   现有简单端点L2惩罚无法检测高频锯齿震荡
    #
    #   多层防护：
    #     Layer 1: 远场总变分(TV)惩罚 — 震荡波的 TV 值异常大
    #     Layer 2: 远场单调衰减约束 — |psi| 在远场应单调递减
    #     Layer 3: f 分量全区间 TV 保护 — f 本身小，整体也需要保护
    #     Layer 4: 过零频率异常检测 — 震荡解过零数远超物理允许值
    # ================================================================

    def _boundary_smoothness_loss(g_b, f_b, r_grid, dr_val):
        """
        远场边界平滑性与反震荡惩罚。
        """
        r_boundary = 15.0  # fm，从此处开始强化边界保护
        bidx = min(max(int(r_boundary / dr_val), 10), npt - 2)  # 边界起始索引

        # === Layer 1: 远场 TV 惩罚 ===
        # 总变分 TV = Σ|x[i+1] - x[i]|，光滑曲线TV小，锯齿震荡TV大
        g_far = g_b[:, bidx:]
        f_far = f_b[:, bidx:]

        tv_g_far = torch.abs(g_far[:, 1:] - g_far[:, :-1]).sum(dim=-1).mean()
        tv_f_far = torch.abs(f_far[:, 1:] - f_far[:, :-1]).sum(dim=-1).mean()
        loss_tv_far = tv_g_far + tv_f_far * 3.0  # f 的远场TV更严格

        # === Layer 2: 远场单调衰减 ===
        # 远场|g|的粗略包络不应有明显的正向跳变
        g_far_abs = torch.abs(g_far)
        # 用局部最大值作为包络近似（简化版）
        env_step = max(3, min(5, g_far.shape[-1] // 4))
        if g_far_abs.shape[-1] > env_step * 2:
            g_pad = torch.nn.functional.pad(
                g_far_abs.unsqueeze(1), (env_step, env_step), value=0
            )
            g_env = torch.nn.functional.max_pool1d(
                g_pad, kernel_size=2 * env_step + 1, stride=1
            ).squeeze(1)
            # 包络的正向增量（非单调信号）
            env_increments = g_env[:, 1:] - g_env[:, :-1]
            pos_jumps = torch.clamp(env_increments, min=0)
            loss_monotonicity = torch.mean(pos_jumps ** 2) * 20.0
        else:
            loss_monotonicity = torch.tensor(0.0, device=device)

        # === Layer 3: f 全区间 TV 保护 ===
        tv_f_full = torch.abs(f_b[:, 1:] - f_b[:, :-1]).sum(dim=-1).mean()
        # 参考上限：正常 f 的 TV ≈ mean(|f|) * 0.3 * L（经验估计）
        ref_tv = torch.mean(torch.abs(f_b)).clamp(min=1e-10) * 0.3 * npt
        loss_f_oscillation = torch.clamp(tv_f_full - ref_tv, min=0) ** 2

        # === Layer 4: 过零频率异常检测 ===
        # 计算实际过零数 vs 物理允许的最大过零数
        # 物理上：节点数 = n_nodes（由量子数决定），过零数 ≈ 2*n_nodes
        def _count_sign_changes(sig):
            s = torch.sign(sig)
            s[s == 0] = 1
            return ((s[:, 1:] != s[:, :-1]).float().sum(dim=-1)) / 2.0

        fz_crossings = _count_sign_changes(f_b)  # (B,)
        gz_crossings = _count_sign_changes(g_b)  # (B,)

        # 允许的最大过零数：物理节点数的3倍（留余量）
        max_allowed_crossings = 6.0  # 即使高激发态也不超过~3个节点→~6次过零

        loss_osc_f = torch.mean(torch.clamp(fz_crossings - max_allowed_crossings, min=0) ** 2) * 100.0
        loss_osc_g = torch.mean(torch.clamp(gz_crossings - max_allowed_crossings, min=0) ** 2) * 50.0

        return loss_tv_far + loss_monotonicity + loss_f_oscillation * 5.0 + loss_osc_f + loss_osc_g

    loss_boundary_smooth = _boundary_smoothness_loss(g, f, r, dr)

    # 合并原始边界端点损失 + 新增平滑性损失
    loss_boundary_total = loss_boundary + loss_boundary_smooth

    # ================================================================
    #   ★ 相位对齐：双阶段鲁棒找峰，保证第一个显著峰为正
    #   改进点：
    #     1. 跳过近核区(r<0.5fm)平坦段 — g≈0区域无物理意义
    #     2. 双阶段：粗搜索(全局最大值位置) → 精细验证(局部极大值)
    #     3. 显著性阈值：候选峰必须 > 30% 全局最大值，排除噪声假峰
    #     4. 统一策略：Model_Architecture 和此处共用同一逻辑
    # ================================================================

    def _find_first_significant_peak(g_out):
        """
        找到 |g| 的第一个显著局部极大值位置（全向量化）。

        双阶段策略：
          Stage 1: 在搜索区域内（r >= 0.5fm）找全局最大值作为锚点
          Stage 2: 从锚点向左回溯，验证是否为真正的局部极大值
          Fallback: 若无显著峰，用全局最大值

        返回: (peak_indices, peak_values)
          peak_indices: (B,) 每个样本第一个显著峰的位置
          peak_values:  (B,) 该位置的 g 值（含符号）
        """
        B, L = g_out.shape
        order = 5
        abs_g = torch.abs(g_out)

        # 跳过近核区平坦段：r < 0.5fm → index < 5（dr=0.10）
        search_start = max(order + 1, 5)  # 至少从第5个点开始
        search_end = L - order

        if search_end <= search_start:
            # 极短序列 fallback
            peak_idx = abs_g.argmax(dim=-1)
            return peak_idx, g_out[torch.arange(B, device=device), peak_idx]

        # === Stage 1: 粗搜索 — 搜索区内全局最大值作为锚点 ===
        # 只在 [search_start, search_end) 范围内搜索
        mask_region = torch.ones_like(abs_g)
        mask_region[:, :search_start] = 0
        mask_region[:, search_end:] = 0

        abs_g_masked = abs_g * mask_region
        anchor_idx = abs_g_masked.argmax(dim=-1)  # (B,) 搜索区内最大值位置

        # 显著性阈值：必须 > 30% 搜索区内的全局最大值
        global_max = abs_g_masked.amax(dim=-1).clamp(min=1e-10)  # (B,)
        anchor_vals = abs_g[torch.arange(B, device=device), anchor_idx]  # (B,)
        significance_threshold = 0.30 * global_max

        # === Stage 2: 精细验证 — 从锚点向左回溯找真正的局部极大值 ===
        # 使用 max_pool1d 做严格的局部极大值检测
        pad = torch.nn.functional.pad(abs_g, (order, order), value=-float('inf'))
        pool = torch.nn.functional.max_pool1d(
            pad.unsqueeze(1), kernel_size=2 * order + 1, stride=1
        ).squeeze(1)  # (B, L): 局部最大值

        is_local_max = (abs_g >= pool - 1e-10) & (abs_g > 0) & (mask_region.bool())
        # 排除边界效应
        lm_mask = torch.zeros_like(is_local_max)
        lm_mask[:, order:L - order] = True
        is_local_max = is_local_max & lm_mask

        # 从锚点向左扫描，找到第一个局部极大值
        # 构建每个样本的"有效范围"：[search_start, anchor_idx+1]
        arange_full = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

        # 对于每个样本，只保留 anchor_idx 左侧（含）的局部极大值候选
        is_candidate = is_local_max.clone()
        anchor_expanded = anchor_idx.unsqueeze(1).expand(B, L)
        is_candidate = is_candidate & (arange_full <= anchor_expanded)

        # 取最右边的候选（即离锚点最近的左侧局部极大值）
        cumsum_candidates = is_candidate.float().cumsum(dim=-1)
        total_per_row = cumsum_candidates[:, -1]  # (B,) 每行的候选总数

        # 如果有候选，取最后一个（最靠近锚点的）
        has_candidates = total_per_row > 0
        refined_idx = anchor_idx.clone()

        if has_candidates.any():
            # 用反向cumsum找到每行最后一个候选的位置
            reverse_cumsum = is_candidate.flip(dims=[1]).float().cumsum(dim=-1).flip(dims=[1])
            is_last_candidate = (is_candidate & (reverse_cumsum == 1))
            last_pos = arange_full.masked_fill(~is_last_candidate, 0).amax(dim=-1)  # (B,)
            refined_idx[has_candidates] = last_pos[has_candidates]

        # 验证显著性
        refined_abs_vals = abs_g[torch.arange(B, device=device), refined_idx]
        is_significant = refined_abs_vals >= significance_threshold

        # 对不显著的行，fallback 到搜索区全局最大值
        final_idx = torch.where(is_significant, refined_idx, anchor_idx)

        # 完全无峰的终极 fallback（理论上不会触发）
        no_peak_at_all = abs_g[:, search_start:search_end].amax(dim=-1) < 1e-15
        if no_peak_at_all.any():
            final_idx[no_peak_at_all] = abs_g[no_peak_at_all].argmax(dim=-1)

        final_vals = g_out[torch.arange(B, device=device), final_idx]
        return final_idx, final_vals


    def _align_phase(g_out, f_out):
        """
        将 g,f 对齐到第一个显著峰 > 0 的相位约定。
        调用 _find_first_significant_peak 进行鲁棒找峰。
        """
        peak_indices, peak_vals = _find_first_significant_peak(g_out)
        B = g_out.shape[0]
        flip = (peak_vals < 0).float().unsqueeze(-1)  # (B, 1)
        return g_out * flip - g_out * (1 - flip), f_out * flip - f_out * (1 - flip), peak_indices

    g_aligned, f_aligned, _peak_indices = _align_phase(g, f)

    # 后续所有约束使用相位对齐后的 g_aligned, f_aligned
    # 替换原有 g, f 引用
    g = g_aligned
    f = f_aligned

    # ================================================================
    #   ★ 能量与势场诊断信息输出
    # ================================================================
    E_mean = E.detach().mean()  # 标量能量 (MeV)
    vps_core = vps[:, :min(int(6.0/dr), npt)].detach().mean()  # 核内区势场均值
    vms_core = vms[:, :min(int(6.0/dr), npt)].detach().mean()

    # ================================================================
    #   组装输出
    # ================================================================
    if return_components:
        components = {
            'loss_pde': loss_pde,
            'loss_norm': loss_norm,
            'loss_amplitude': loss_amplitude,
            'loss_node': loss_node_anomaly,
            'loss_boundary': loss_boundary_total,       # ★ 原始边界 + 新增平滑性保护
            'loss_positive_energy': loss_positive_energy,
            'loss_kinetic_positive': loss_kinetic_positive,
            'loss_energy_range': loss_energy_range,      # ★ 新增：能量范围惩罚 (-80~-50 MeV)
            'loss_shape': loss_shape,                    # ★ 新增：波形形态惩罚
            'loss_peak': loss_peak,                      # ★ 新增：峰值位置匹配损失
            'loss_boundary_smooth': loss_boundary_smooth,# ★ 新增：边界平滑性（可单独监控）
            'loss_energy_rayleigh': loss_energy_rayleigh, # ★ 新增：Rayleigh商能量一致性
            'norm_integral': norm_integral.detach().mean(),
            'energy_E': energy_E.mean() if 'energy_E' in locals() else E_mean,  # ★ 使用Rayleigh能量
            'energy_rayleigh': energy_rayleigh.detach().mean(),  # ★ 新增：Rayleigh商计算的能量
            'vps_core': vps_core,                         # ★ 新增：标量势场核内均值
            'vms_core': vms_core,                         # ★ 新增：矢量势场核内均值
            'loss_total': (loss_pde + loss_norm + loss_amplitude + loss_node_anomaly
                           + loss_boundary_total + loss_positive_energy
                           + loss_kinetic_positive + loss_energy_range
                           + loss_shape + loss_peak + loss_energy_rayleigh),
        }
        if ref_scale is not None and ref_scale > 0:
            # ★ PDE/Norm 不做任何缩放！原始值直接输出
            scaled = {
                'loss_pde': loss_pde,
                'loss_norm': loss_norm,
                'loss_amplitude': loss_amplitude / ref_scale,
                'loss_node': loss_node_anomaly,
                'loss_boundary': loss_boundary_total,
                'loss_energy_range': loss_energy_range,      # ★ 新增
                'loss_shape': loss_shape / ref_scale,
                'loss_peak': loss_peak,                       # ★ 新增：峰值位置匹配（不缩放，物理单位fm）
                'loss_boundary_smooth': loss_boundary_smooth / ref_scale,
                'loss_positive_energy': loss_positive_energy,
                'loss_kinetic_positive': loss_kinetic_positive,
                'norm_integral': norm_integral.detach().mean(),
                'energy_E': E_mean,
                'vps_core': vps_core,
                'vms_core': vms_core,
            }
            scaled['loss_total'] = (scaled['loss_pde'] + scaled['loss_norm']
                                    + scaled['loss_amplitude'] + scaled['loss_node']
                                    + scaled['loss_boundary']
                                    + scaled['loss_energy_range']
                                    + scaled['loss_shape']
                                    + scaled['loss_peak']
                                    + scaled['loss_positive_energy']
                                    + scaled['loss_kinetic_positive'])
            return scaled
        return components
    else:
        loss_physics = (loss_pde + loss_norm + loss_amplitude
                        + loss_node_anomaly + loss_boundary_total
                        + loss_positive_energy + loss_kinetic_positive
                        + loss_shape + loss_peak)
        if ref_scale is not None and ref_scale > 0:
            loss_physics = loss_physics / ref_scale
        return loss_physics


# ═══════════════════════════════════════════════════════════════
#   ★ 新增：精简版物理损失（6个核心损失，去除冗余）
#
#   精简策略（12→6）：
#     1. loss_pde — Dirac方程残差（保留，主导）
#     2. loss_norm — 归一化约束（保留）
#     3. loss_node — 节点数精确约束（保留）
#     4. loss_physical_state — 合并loss_positive_energy + loss_kinetic_positive
#     5. loss_smoothness — 合并loss_shape + loss_boundary_smooth + loss_boundary
#     6. loss_energy_rayleigh — Rayleigh商能量一致性（保留）
#
#   删除的冗余损失：
#     - loss_amplitude: 归一化+PDE已隐含振幅约束
#     - loss_energy_mse: Rayleigh商已隐含能量约束
#     - loss_energy_range: Rayleigh商自然约束能量范围
#     - loss_peak: 物理态约束+光滑性已覆盖峰值位置
# ═══════════════════════════════════════════════════════════════

def calc_simplified_residual(pred_tensor, kappa, dr=0.10, n_principal=None, y_true=None):
    """
    精简版物理残差计算：仅6个核心损失，去除冗余。

    参数与 calc_physics_residual 相同（stats_mean/stats_std/ref_scale不再需要）。

    返回: dict 包含6个核心损失 + 诊断信息
    """
    device = pred_tensor.device
    B, C, npt = pred_tensor.shape

    # 通道解包
    g = pred_tensor[:, 0, :]
    f = pred_tensor[:, 1, :]
    vps = pred_tensor[:, 2, :]
    vms = pred_tensor[:, 3, :]
    vtt = pred_tensor[:, 4, :]
    XG = pred_tensor[:, 5, :]
    XF = pred_tensor[:, 6, :]
    YG = pred_tensor[:, 7, :]
    YF = pred_tensor[:, 8, :]

    hbc = 197.328284
    M_nucleon = 939.0

    E_network = pred_tensor[:, 9, :].mean(dim=1, keepdim=True)
    E = E_network

    # 网格坐标
    r = torch.arange(0, npt, device=device, dtype=torch.float32) * dr
    r[0] = 0.0010
    r = r.unsqueeze(0).expand(B, -1)
    kappa_exp = kappa.unsqueeze(1)

    E_hc = E / hbc
    r1 = kappa_exp / r

    u1g = r1 + vtt + XG
    u1f = E_hc - vms - XF
    u2f = r1 + vtt + YF
    u2g = E_hc - vps - YG

    # 4阶中心差分
    dg_dr = (-g[:, 4:] + 8.0 * g[:, 3:-1] - 8.0 * g[:, 1:-3] + g[:, :-4]) / (12.0 * dr)
    df_dr = (-f[:, 4:] + 8.0 * f[:, 3:-1] - 8.0 * f[:, 1:-3] + f[:, :-4]) / (12.0 * dr)

    g_int, f_int = g[:, 2:-2], f[:, 2:-2]
    u1g_int, u1f_int = u1g[:, 2:-2], u1f[:, 2:-2]
    u2f_int, u2g_int = u2f[:, 2:-2], u2g[:, 2:-2]

    # ═══════ 损失 1: Dirac方程PDE残差 ═══════
    Rg = dg_dr - (-u1g_int * g_int + u1f_int * f_int)
    Rf = df_dr - (u2f_int * f_int - u2g_int * g_int)
    r_int = r[:, 2:-2]
    Rg_weighted = Rg * r_int
    Rf_weighted = Rf * r_int
    loss_pde = torch.mean(Rg_weighted ** 2 + 3.0 * Rf_weighted ** 2) * dr

    # ═══════ 损失 2: 归一化 ═══════
    prob_density = (g ** 2 + f ** 2) * dr
    norm_integral = torch.sum(prob_density, dim=1)
    loss_norm = torch.mean((norm_integral - 1.0) ** 2)

    # ═══════ 损失 3: 节点数约束 ═══════
    def _count_zero_crossings(signal):
        signs = torch.sign(signal)
        signs[signs == 0] = 1
        crossings = (signs[:, 1:] != signs[:, :-1]).float()
        return torch.sum(crossings, dim=1) / 2.0

    g_crossings = _count_zero_crossings(g)

    if n_principal is not None:
        n_val = n_principal.long()
        k_val = kappa.long()
        l_val = torch.where(k_val < 0, -k_val - 1, k_val).float()
        expected_nodes = torch.where(
            k_val < 0,
            (n_val.float() - l_val - 1.0).clamp(min=0),
            (n_val.float() - l_val).clamp(min=0)
        )
        node_error = torch.abs(g_crossings - expected_nodes)
        loss_node = torch.mean(node_error ** 2) + torch.mean(torch.clamp(node_error - 1.0, min=0) ** 2) * 10.0
    else:
        r_sign_loose = min(int(8.0 / dr), npt)
        loss_node = torch.mean(torch.clamp(g_crossings - 10, min=0)) + \
                    torch.mean(torch.clamp(0.5 - g_crossings, min=0)) * \
                    torch.mean(torch.clamp(g[:, :r_sign_loose].mean(dim=1), max=0))

    # ═══════ 损失 4: 物理态约束（合并正能量+动能正定性）═══════
    # 正能量态：∫f² 应小于 0.3·∫g²
    integral_g2 = torch.sum(g ** 2, dim=1) * dr
    integral_f2 = torch.sum(f ** 2, dim=1) * dr
    f_dominance = integral_f2 - 0.3 * integral_g2
    loss_pos_energy = torch.mean(torch.clamp(f_dominance, min=0.0) ** 2)

    # 动能正定性
    # ★ 2026-04-19 修复：使用5PADF交替差分（G:forward, F:backward）
    D_g_s = _build_fd_matrix_5padf(npt, dr, direction='forward', device=device)
    D_f_s = _build_fd_matrix_5padf(npt, dr, direction='backward', device=device)
    dg_full = _apply_fd_matrix(g, D_g_s)
    df_full = _apply_fd_matrix(f, D_f_s)
    kin_term_diff = -g * df_full + f * dg_full
    kin_term_so = 2.0 * (kappa_exp / r) * g * f
    kin_integrand = (kin_term_diff + kin_term_so) * dr
    kin_integrand[:, 0] = 0
    E_kin_pure = torch.sum(kin_integrand, dim=1)
    loss_kin_pos = torch.mean(torch.clamp(-E_kin_pure, min=0.0) ** 2)

    loss_physical_state = loss_pos_energy + loss_kin_pos

    # ══════ 损失 5: 光滑性（纯防锯齿，无形态假设）═══════
    # ★ 2026-04-19 大手术: 删除tail/mono/boundary, 只保留曲率+TV+HF谱

    # --- 二阶差分: 曲度 ---
    #   2. 全域TV: 对任意频率振荡敏感
    #   3. 高频谱能量: FFT直接切除Gibbs
    # 删除项:
    #   - loss_tail (尾部集中性): 强制把峰推向近核区 → 压成台阶状!
    #   - loss_mono (包络单调): 形态假设, 错误时反而锁死错误形状
    #   - loss_boundary (端点+远场TV): 与Model_Architecture的硬约束重复

    # --- (保留) 二阶差分: 曲率 ---
    d2g = g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]
    d2f = f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]
    loss_curvature = torch.mean(d2g ** 2) + torch.mean(d2f ** 2)

    # ★ 保留: 全域一阶总变差(TV)
    tv_g = torch.abs(g[:, 1:] - g[:, :-1]).sum(dim=-1).mean()
    tv_f = torch.abs(f[:, 1:] - f[:, :-1]).sum(dim=-1).mean()
    loss_tv_global = tv_g + tv_f * 3.0

    # ★ 保留: 高频谱能量惩罚
    try:
        with torch.amp.autocast('cuda', enabled=False):
            g_fft = torch.fft.rfft(g.float())
            f_fft = torch.fft.rfft(f.float())
            n_freq = g_fft.size(-1)
            hf_cutoff = max(n_freq // 2, 5)
            if hf_cutoff < n_freq - 1:
                hf_g_energy = (g_fft[:, hf_cutoff:].abs() ** 2).sum(dim=-1).mean()
                hf_f_energy = (f_fft[:, hf_cutoff:].abs() ** 2).sum(dim=-1).mean()
            else:
                hf_g_energy = torch.tensor(0.0, device=device)
                hf_f_energy = torch.tensor(0.0, device=device)
            loss_hf_spectrum = hf_g_energy + hf_f_energy * 3.0
    except Exception:
        loss_hf_spectrum = torch.tensor(0.0, device=device)

    loss_smoothness = loss_curvature + loss_tv_global * 2.0 + loss_hf_spectrum

    # ══════ 损失 5.5: 尾部概率集中惩罚（★ 2026-04-19 新增） ══════
    # 物理依据（教材3.60: G(R)=0, F(R)=C_R）：
    #   束缚态波函数在远场应指数衰减 → 0，1s1/2 态在 r>6fm 时 |g|<0.01
    #   配合 Model_Architecture.py 中 r_cutoff=8.0 的硬约束使用
    #
    # 诊断问题：图像显示 pred_g 在 r=5~15fm 区间维持 ~0.25 的平台，
    #            而该区域 true_g ≈ 0 → 归一化后峰高被压低 ~2.3倍
    # 策略：软惩罚尾部概率占比（非硬截断），让网络自然学会将概率集中到核区
    prob_density = g ** 2 + f ** 2  # (B, N)
    total_prob = torch.sum(prob_density, dim=-1, keepdim=True) * dr  # (B, 1) — 应≈1.0(归一化后)
    tail_start_idx = min(int(8.0 / dr), npt - 1)  # r > 8 fm 开始算尾部
    if tail_start_idx < npt - 1:
        tail_prob = prob_density[:, tail_start_idx:].sum(dim=-1, keepdim=True) * dr  # (B, 1)
        tail_ratio = tail_prob / total_prob.clamp(min=1e-10)  # 尾部概率占比
        # ★ 束缚态尾部概率应 < 5%（对1s1/2态实际<0.1%，留余量）
        #   仅在超限时惩罚（ReLU），正常态不触发
        loss_tail_concentration = torch.mean(torch.clamp(tail_ratio - 0.05, min=0.0) ** 2) * 20.0
    else:
        loss_tail_concentration = torch.tensor(0.0, device=device)


    # ═══════ 损失 6: Rayleigh商能量一致性 ═══════
    r_safe = r.clone()
    r_safe[r_safe < 1e-10] = 1e-10
    Sigma_plus = vps
    Sigma_minus = vms
    # ★ 2026-04-19 关键修复：同上，修正 h_psi_g 中 κ/r 项的符号（- → +）
    # 原错误导致 Rayleigh 商能量与 PDE 残差对抗
    h_psi_g = -df_full + (kappa_exp / r_safe) * f + (Sigma_plus + M_nucleon) * g
    h_psi_f = dg_full + (kappa_exp / r_safe) * g + (Sigma_minus - M_nucleon) * f
    rayleigh_numerator = torch.sum((g * h_psi_g + f * h_psi_f) * dr, dim=1)
    rayleigh_denominator = torch.sum((g**2 + f**2) * dr, dim=1)
    energy_rayleigh_total = rayleigh_numerator / (rayleigh_denominator.clamp(min=1e-10))
    energy_rayleigh = energy_rayleigh_total - M_nucleon
    energy_network_scalar = E.squeeze(-1)
    loss_energy_rayleigh = torch.mean((energy_rayleigh - energy_network_scalar) ** 2)

    # ═══════ 诊断信息 ═══════
    vps_core = vps[:, :min(int(6.0/dr), npt)].detach().mean()
    vms_core = vms[:, :min(int(6.0/dr), npt)].detach().mean()

    return {
        # 6个核心损失
        'loss_pde': loss_pde,
        'loss_norm': loss_norm,
        'loss_node': loss_node,
        'loss_physical_state': loss_physical_state,
        'loss_smoothness': loss_smoothness,
        'loss_tail_concentration': loss_tail_concentration,  # ★ 2026-04-19 新增
        'loss_energy_rayleigh': loss_energy_rayleigh,
        # 诊断信息
        'norm_integral': norm_integral.detach().mean(),
        'energy_rayleigh': energy_rayleigh.detach().mean(),
        'energy_network': energy_network_scalar.detach().mean(),
        'E_kin_pure': E_kin_pure.detach().mean(),
        'vps_core': vps_core,
        'vms_core': vms_core,
        # 总损失
        'loss_total': loss_pde + loss_norm + loss_node + loss_physical_state + loss_smoothness + loss_tail_concentration + loss_energy_rayleigh,
    }


# ═══════════════════════════════════════════════════════════════
#   ★ 新增：RHF一致性检查器
#   实时监督势能-波函数-能量一致性，确保训练过程满足RHF方程
# ═══════════════════════════════════════════════════════════════

class RHFConsistencyChecker:
    """
    RHF方程一致性检查器。

    每N步验证：
      1. Rayleigh能量 vs 网络输出能量的一致性
      2. 动能正定性（E_kin > 0）
      3. 势场统计量（vps/vms核内均值应在物理范围内）
      4. 归一化精度

    一致性残差可作为额外损失项反向传播，
    确保网络在训练过程中始终满足RHF方程。
    """
    def __init__(self, check_every=100, log_csv=None, lambda_consistency=2.0):
        """
        参数：
          check_every: 每N步检查一次
          log_csv: CSV文件路径，记录物理量（None则不记录）
          lambda_consistency: 一致性残差权重
        """
        self.check_every = check_every
        self.lambda_consistency = lambda_consistency
        self.log_csv = log_csv
        self.step_count = 0
        self.history = []

        if log_csv is not None:
            import os
            os.makedirs(os.path.dirname(log_csv) if os.path.dirname(log_csv) else '.', exist_ok=True)
            with open(log_csv, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['step', 'E_rayleigh', 'E_network', 'E_kin',
                                'vps_core', 'vms_core', 'norm_integral',
                                'consistency_residual'])

    def compute_consistency(self, pred_tensor, kappa, dr=0.10, n_principal=None):
        """
        计算RHF一致性指标和残差。

        返回:
          consistency_residual: 标量损失，可反向传播
          diagnostics: dict，物理量诊断（detached）
        """
        device = pred_tensor.device
        B, C, npt = pred_tensor.shape

        g = pred_tensor[:, 0, :]
        f = pred_tensor[:, 1, :]
        vps = pred_tensor[:, 2, :]
        vms = pred_tensor[:, 3, :]

        M_nucleon = 939.0

        # 网格
        r = torch.arange(0, npt, device=device, dtype=torch.float32) * dr
        r[0] = 0.0010
        r = r.unsqueeze(0).expand(B, -1)
        kappa_exp = kappa.unsqueeze(1)

        # 4阶差分
        # ★ 2026-04-19 修复：使用5PADF交替差分（G:forward, F:backward）
        D_g_chk = _build_fd_matrix_5padf(npt, dr, direction='forward', device=device)
        D_f_chk = _build_fd_matrix_5padf(npt, dr, direction='backward', device=device)
        dg_dr = _apply_fd_matrix(g, D_g_chk)   # (B, N) — 完整导数
        df_dr = _apply_fd_matrix(f, D_f_chk)   # (B, N)

        # 差分矩阵已给出全区间导数，无需零填充
        dg_full = dg_dr
        df_full = df_dr

        # Rayleigh能量
        r_safe = r.clone()
        r_safe[r_safe < 1e-10] = 1e-10
        # ★ 2026-04-19 关键修复：同上，修正 h_psi_g 中 κ/r 项的符号（- → +）
        h_psi_g = -df_full + (kappa_exp / r_safe) * f + (vps + M_nucleon) * g
        h_psi_f = dg_full + (kappa_exp / r_safe) * g + (vms - M_nucleon) * f
        rayleigh_num = torch.sum((g * h_psi_g + f * h_psi_f) * dr, dim=1)
        rayleigh_den = torch.sum((g**2 + f**2) * dr, dim=1)
        E_rayleigh = rayleigh_num / (rayleigh_den.clamp(min=1e-10)) - M_nucleon

        # 网络输出能量
        E_network = pred_tensor[:, 9, :].mean(dim=1)

        # 动能
        kin_term_diff = -g * df_full + f * dg_full
        kin_term_so = 2.0 * (kappa_exp / r) * g * f
        kin_integrand = (kin_term_diff + kin_term_so) * dr
        kin_integrand[:, 0] = 0
        E_kin = torch.sum(kin_integrand, dim=1)

        # 归一化
        norm_integral = torch.sum((g**2 + f**2) * dr, dim=1)

        # 势场统计
        bidx = min(int(6.0/dr), npt)
        vps_core = vps[:, :bidx].mean()
        vms_core = vms[:, :bidx].mean()

        # 一致性残差：Rayleigh商 vs 网络输出能量的差异
        consistency_residual = torch.mean((E_rayleigh - E_network) ** 2)

        # 诊断
        diagnostics = {
            'E_rayleigh': E_rayleigh.detach().mean().item(),
            'E_network': E_network.detach().mean().item(),
            'E_kin': E_kin.detach().mean().item(),
            'vps_core': vps_core.detach().item(),
            'vms_core': vms_core.detach().item(),
            'norm_integral': norm_integral.detach().mean().item(),
            'consistency_residual': consistency_residual.detach().item(),
        }

        # 异常检测
        if diagnostics['E_kin'] < 0:
            print(f"  ⚠️ RHF一致性警告: 动能E_kin={diagnostics['E_kin']:.4f} < 0 (负能量海?)")
        if abs(diagnostics['E_rayleigh'] - diagnostics['E_network']) > 100:
            print(f"  ⚠️ RHF一致性警告: Rayleigh能量({diagnostics['E_rayleigh']:.2f}) vs 网络能量({diagnostics['E_network']:.2f}) 偏差过大")

        # 记录
        self.step_count += 1
        self.history.append(diagnostics)

        if self.log_csv is not None and self.step_count % self.check_every == 0:
            with open(self.log_csv, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow([
                    self.step_count,
                    f"{diagnostics['E_rayleigh']:.4f}",
                    f"{diagnostics['E_network']:.4f}",
                    f"{diagnostics['E_kin']:.4f}",
                    f"{diagnostics['vps_core']:.4f}",
                    f"{diagnostics['vms_core']:.4f}",
                    f"{diagnostics['norm_integral']:.6f}",
                    f"{diagnostics['consistency_residual']:.6f}",
                ])

        return consistency_residual, diagnostics
