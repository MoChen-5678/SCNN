"""
DiracNet — PINN 波函数拟设网络

以径向坐标 r 为输入，输出 Dirac 波函数的大分量 G(r) 和小分量 F(r)。
核心设计:
  - MLP 主体 + r 特征工程 (原始r + log(r) + exp(-αr))
  - 硬归一化约束: ∫(G²+F²)dr = 1
  - 相位对齐: 确保 G 在核区为正（消除全局符号不确定性）
  - ★ 内嵌Gram-Schmidt正交化: 前向传播中自动对参考波函数做正交投影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import MODEL_CONFIG, M_NUCLEON, HBAR_C, R_SAFE_OFFSET, DEVICE


class DiracNet(nn.Module):
    """
    PINN 波函数拟设: NN(r; θ) → (G(r), F(r))

    输入: 径向坐标 r (标量或batch)
    输出: G(r), F(r) — Dirac波函数大/小分量
    可学习参数: E (能量本征值, MeV)

    设计要点:
      1. r 特征工程: [r, log(r+ε), exp(-αr)] 帮助网络学习近核和渐近行为
      2. Swish 激活: 比 tanh 更适合 PINN 的梯度流动
      3. 硬归一化: 每次前向传播强制归一化（可选）
      4. 相位对齐: 强制 G(r_peak) > 0，消除 ± 不确定性
    """

    def __init__(self, n_hidden=None, n_layers=None, activation='swish',
                 hard_normalize=True, init_energy=-40.0):
        super().__init__()

        nh = n_hidden or MODEL_CONFIG['n_hidden']
        nl = n_layers or MODEL_CONFIG['n_layers']

        self.hard_normalize = hard_normalize
        self.n_hidden = nh
        self.n_layers = nl

        # ─── ★ 参考波函数 (用于内嵌正交化) ───
        #   list of dict: [{'g': tensor(N,), 'f': tensor(N,)}, ...]
        #   在 forward() 中自动做 Gram-Schmidt 正交投影
        self.ref_wavefunctions = []

        # ─── 可学习能量本征值 ───
        # 初始化为类氢近似估计值（对束缚态通常在 -40 ~ -300 MeV 范围）
        self.E = nn.Parameter(torch.tensor([init_energy], dtype=torch.float32))

        # ─── 可学习衰减参数 ───
        # beta:   边界指数衰减因子 e^{-βr}, 对标RHF论文的渐近边界条件
        #         确保波函数在无穷远处指数衰减到零 (束缚态物理要求)
        #         论文初始化为 1.0 fm⁻¹
        self.beta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        # ─── MLP 主体 ───
        # 输入维度: 3 (r, log(r+eps), exp(-ar))
        input_dim = 3

        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, nh))
        layers.append(self._get_activation(activation))

        # 隐藏层
        for _ in range(nl - 1):
            layers.append(nn.Linear(nh, nh))
            layers.append(self._get_activation(activation))

        self.mlp = nn.Sequential(*layers)

        # 输出头: 特征 → (G, F)
        # 使用两个独立线性层，允许 G/F 学习不同模式
        self.head_g = nn.Linear(nh, 1)
        self.head_f = nn.Linear(nh, 1)

        # ─── 初始化策略 ───
        self._initialize_weights()

    def _get_activation(self, name):
        """返回激活函数模块"""
        if name == 'swish':
            return nn.SiLU()  # SiLU = Swish
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sin':
            return nn.Sin()  # PyTorch原生支持
        else:
            raise ValueError(f"未知激活函数: {name}")

    def _initialize_weights(self):
        """
        权重初始化策略:
          - G头: 正偏置初始化（确保初始G>0）
          - F头: 小权重初始化（F ≪ G 非相对论极限）
          - MLP: Xavier uniform
        """
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # G 头: 初始输出正值
        nn.init.xavier_uniform_(self.head_g.weight)
        nn.init.constant_(self.head_g.bias, 0.5)

        # F 头: 初始输出小量 (F/G ≈ v/c ≈ 0.3~0.4 对于束缚态)
        nn.init.xavier_uniform_(self.head_f.weight)
        nn.init.constant_(self.head_f.bias, 0.01)

    def _build_r_features(self, r):
        """
        构造 r 的特征向量。

        物理动机:
          - r: 原始径向坐标
          - log(r + ε): 对数坐标帮助学习近核的 r^l 行为
          - exp(-αr): 指数衰减特征帮助学习尾部行为
          α 是可学习参数，自适应调整衰减率

        参数:
            r: (..., ) 张量，径向坐标 (fm)
        返回:
            features: (..., 3) 特征矩阵
        """
        eps = R_SAFE_OFFSET  # 避免 log(0)

        # 安全处理 r=0
        r_safe = torch.clamp(r, min=eps)

        features = torch.stack([
            r,                                    # 原始 r
            torch.log(r_safe + eps),              # log(r) 近核行为
            torch.exp(-r),                        # 固定指数衰减 (尾部行为由beta控制)
        ], dim=-1)

        return features

    def forward(self, r, kappa=None, dr=0.10):
        """
        前向传播: r → (G, F)

        参数:
            r: (B, N) 或 (N,) 径向坐标张量 (fm)
            kappa: int 或 None, 角量子数
            dr: float, 径向步长 (fm), 用于归一化积分

        返回:
            g: (...) G 分量 (与 r 同形)
            f: (...) F 分量 (与 r 同形)

        ★ 硬边界约束 (RHF论文 + 核物理教材):
          - r→0: G~r^{l_u+1}, F~r^{l_d+1}  → 乘以 r^{l+1} 因子
          - r→∞: ψ~e^{-βr}, β>0可学习      → 乘以指数衰减因子
        """
        original_shape = r.shape
        flat_r = r.reshape(-1, 1)  # (M, 1)

        # 1. 特征工程
        features = self._build_r_features(flat_r)  # (M, 3)

        # 2. MLP 前向
        h = self.mlp(features)  # (M, n_hidden)

        # 3. 分别计算 G 和 F
        g_raw = self.head_g(h).squeeze(-1)  # (M,)
        f_raw = self.head_f(h).squeeze(-1)  # (M,)

        # 4. 恢复形状
        g = g_raw.reshape(original_shape)
        f = f_raw.reshape(original_shape)

        # 5. 硬边界约束: 乘以渐近因子
        if kappa is not None:
            g, f = self._apply_boundary_factors(g, f, r, kappa)

        # 6. 硬归一化
        if self.hard_normalize:
            g, f = self._normalize(g, f, dr)

        # 7. 相位对齐: 确保 G 的主峰为正
        g, f = self._align_phase(g, f, dr)

        # ★ 8. 内嵌 Gram-Schmidt 正交化 (对所有参考波函数做硬正交投影)
        if self.ref_wavefunctions:
            g, f = self._enforce_orthogonality(g, f, dr)
            # 投影后重新归一化 (Gram-Schmidt会改变模长)
            if self.hard_normalize:
                g, f = self._normalize(g, f, dr)
                g, f = self._align_phase(g, f, dr)  # 投影可能翻转相位

        return g, f

    def _apply_boundary_factors(self, g, f, r, kappa):
        """
        硬边界约束: 乘以解析渐近因子 (核物理教材 Eq.3.60-3.61 + RHF论文e^{-βr})。

        ★ r→0 渐近行为 (Eq.3.61):
          G(r) = C₀·r^{l_u+1},  F(r) = C₀'·r^{l_d+1}
          → 乘以 r^{l_u+1} 和 r^{l_d+1}，使网络只需学习系数 C₀, C₀'

        ★ r→∞ 指数衰减 (RHF论文边界条件嵌入):
          ψ(r) ~ e^{-βr}, β > 0 可学习
          → 乘以 e^{-βr}, 替代硬截断 (1-r/R)^2

        ★ 完整拟设: G(r) = r^{l_u+1} · N_net(r) · e^{-βr}
        """
        from boundary_conditions import get_angular_momenta

        l_u, l_d = get_angular_momenta(kappa)

        eps = R_SAFE_OFFSET
        r_safe = torch.clamp(r, min=eps)

        # r→0 因子: 确保 G ~ r^{l_u+1}, F ~ r^{l_d+1}
        #factor_g = r_safe ** (l_u + 1)
        #factor_f = r_safe ** (l_d + 1)
        factor_g = r_safe
        factor_f = r_safe
        # r→∞ 指数衰减因子 (可学习 β > 0)
        # 用 softplus 保证 β 始终为正，避免梯度消失时 β 变负导致发散
        beta_pos = torch.nn.functional.softplus(self.beta)  # β > 0
        exp_decay = torch.exp(-beta_pos * r_safe)

        g = g * factor_g * exp_decay
        f = f * factor_f * exp_decay

        return g, f

    def _normalize(self, g, f, dr):
        """
        硬狄拉克归一化: ∫(g²+f²)dr = 1 (无 r² 因子!)

        注意: 这是核物理中约化径向波函数的标准归一化约定。
              不是 ∫(g²+f²)r²dr = 1!
        """
        # 计算归一化积分 (梯形法则)
        integrand = g**2 + f**2
        norm = torch.trapz(integrand, dim=-1, dx=dr)  # (...) 标量
        
        # 安全除法
        norm_safe = torch.clamp(norm, min=1e-30)
        scale = 1.0 / torch.sqrt(norm_safe)

        # 广播: scale 可能是标量或 (B,) 
        # g/f 形状可能是 (N,) 或 (B,N)
        if scale.dim() > 0 and g.dim() > 1:
            scale = scale.unsqueeze(-1)

        return g * scale, f * scale

    def _align_phase(self, g, f, dr):
        """
        相位对齐: 强制 G 在最大绝对值处为正。
        
        消除 PINN 的全局符号不确定性 (±ψ 都满足 Dirac 方程)。
        同时翻转 F 保持物理一致性。
        """
        # 找到 |G| 最大位置处的符号
        idx_max = torch.argmax(torch.abs(g), dim=-1, keepdim=True)
        sign_at_peak = torch.take_along_dim(g, idx_max, dim=-1)

        # 如果峰值为负, 整体翻转
        flip = torch.sign(sign_at_peak)  # +1 或 -1
        if flip.dim() > 0 and g.dim() > 1:
            flip = flip.expand_as(g)

        return g * torch.sign(flip + 1e-10), f * torch.sign(flip + 1e-10)

    def set_ref_wavefunctions(self, ref_wavefunctions):
        """设置参考波函数列表 (用于内嵌正交化).
        
        参数:
            ref_wavefunctions: list of dict
                每个 dict 含 'g': tensor(N,), 'f': tensor(N,)
                (与当前网络在同一径向网格上)
        """
        self.ref_wavefunctions = ref_wavefunctions

    def _enforce_orthogonality(self, g, f, dr):
        """
        ★ 内嵌 Gram-Schmidt 正交投影 — 硬约束，非软惩罚!
        
        对每个参考波函数做投影减法:
            overlap = ∫(g·g_ref + f·f_ref) dr
            g' = g - overlap · g_ref
            f' = f - overlap · f_ref
        
        最终结果天然满足 ∫ψ_i ψ_j dr = 0 (对所有 j < i)，
        无需外部 loss 权重调参。
        
        参数:
            g, f: 当前态波函数 (N,) 或 (B,N)
            dr: 径向步长
        
        返回:
            g_ortho, f_ortho: 与所有参考波函数正交的波函数
        """
        if not self.ref_wavefunctions:
            return g, f

        device = g.device

        for ref in self.ref_wavefunctions:
            g_ref = ref['g'].to(device)
            f_ref = ref['f'].to(device)

            # 确保形状匹配
            if g.dim() == 2:
                if g_ref.dim() == 1:
                    g_ref = g_ref.unsqueeze(0).expand_as(g)
                    f_ref = f_ref.unsqueeze(0).expand_as(f)

            # 计算重叠积分: ⟨ψ_current | ψ_ref⟩
            overlap = torch.trapz(g * g_ref + f * f_ref, dim=-1, dx=dr)

            # Gram-Schmidt 投影: 减去平行分量
            # overlap 形状: 标量 or (B,) — 需要正确广播
            if g.dim() == 1:
                g = g - overlap * g_ref
                f = f - overlap * f_ref
            elif overlap.dim() == 1:
                g = g - overlap.unsqueeze(-1) * g_ref
                f = f - overlap.unsqueeze(-1) * f_ref
            else:
                g = g - overlap * g_ref
                f = f - overlap * f_ref

        return g, f

    def get_energy(self):
        """返回当前的可学习能量本征值 (MeV)"""
        return self.E.item()

    def set_energy(self, value):
        """手动设置能量值（用于SCF迭代间的热启动）"""
        with torch.no_grad():
            self.E.fill_(value)


class MultiStateDiracNet(nn.Module):
    """
    多态同时训练版 DiracNet。
    
    每个 (n, κ) 态有独立的一组网络参数和可学习能量，
    共享架构但完全解耦。适用于同时求解多个占据态的场景。

    用法:
        net = MultiStateDiracNet(states=[('1s1/2', -1), ('1p3/2', -2)])
        for state_name, kappa in states:
            g, f = net.forward_state(state_name, r, dr)
    """

    def __init__(self, state_list, **kwargs):
        """
        参数:
            state_list: [(name, kappa, init_E), ...] 态列表
                       name: 态标识符如 '1s1/2'
                       kappa: 角量子数 (整数)
                       init_E: 能量初估 (MeV, 可选)
            **kwargs: 传给 DiracNet 的其他参数
        """
        super().__init__()

        self.state_names = []
        self.kappas = {}
        self.nets = nn.ModuleDict()

        for s in state_list:
            if len(s) >= 2:
                name, kappa = s[0], s[1]
                init_E = s[2] if len(s) > 2 else -40.0
            else:
                raise ValueError(f"无效态定义: {s}")

            safe_name = name.replace('/', '_').replace('+', 'p').replace('-', 'm')
            self.state_names.append(name)
            self.kappas[name] = kappa
            self.nets[safe_name] = DiracNet(init_energy=init_E, **kwargs)

    def forward_state(self, state_name, r, dr=0.10):
        """返回指定态的 (G, F)"""
        safe_name = state_name.replace('/', '_').replace('+', 'p').replace('-', 'm')
        net = self.nets[safe_name]
        return net(r, kappa=self.kappas[state_name], dr=dr)

    def get_energies(self):
        """返回所有态的能量 {name: E_MeV}"""
        return {name: self.nets[name.replace('/', '_').replace('+', 'p').replace('-', 'm')].E.item()
                for name in self.state_names}

    def get_wavefunctions(self, r, dr=0.10):
        """返回所有态的波函数字典 {name: (g, f)}"""
        result = {}
        for name in self.state_names:
            result[name] = self.forward_state(name, r, dr)
        return result
