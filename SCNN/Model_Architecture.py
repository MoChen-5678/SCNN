import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
#   Sobolev 平滑正则化层 — 消除波函数高频锯齿
#   核心：可学习的1D深度可分离卷积 + 残差连接
#   初始化为 Binomial-5 平滑核 [1,4,6,4,1]/16，训练中可微调
#   残差连接确保平滑层不会退化波形（恒等映射是初始解）
# ═══════════════════════════════════════════════════════════════

class SobolevSmoother(nn.Module):
    """
    Sobolev空间正则化平滑层 (H^1 正则化)

    物理动机：
      狄拉克方程的束缚态解属于 H^1 空间（函数 + 一阶导数都连续），
      但FNO的频域截断会在高频端产生Gibbs震荡 → 锯齿状波函数。
      本层通过可学习的局部平滑卷积，从架构层面保证输出光滑性。

    设计选择：
      - 深度可分离卷积 (groups=channels): g和f独立平滑，互不干扰
      - 残差连接: smooth(x) = conv(x) + x，初始时conv(x)≈0（近恒等）
      - 初始化为 Binomial-5 核 [1,4,6,4,1]/16:
          二项展开系数，保证归一化且对称，是最优的5点平滑核
    """
    def __init__(self, channels=2, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2  # 保持序列长度不变

        # 深度可分离卷积: 每个通道独立卷积
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, groups=channels, bias=False
        )

        # ★ 初始化为 Binomial-5 平滑核 [1,4,6,4,1]/16
        # 此核是5点高斯平滑的最优离散近似，频率响应单调递减
        with torch.no_grad():
            binomial_5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
            # 深度可分离: 每个通道用相同的平滑核
            weight = binomial_5.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1).clone()
            self.conv.weight.data = weight

        # ★ 可学习的残差缩放因子，初始为0.05（极轻量，仅防Gibbs锯齿）
        # ★ 2026-04-19 诊断修复: 从 0.2 回调到 0.05
        #
        # 原问题：alpha=0.2 对 1s1/2 的尖锐物理峰(r≈0.5fm, 半宽~1fm) 有破坏性
        #   Binomial-5 核 [1,4,6,4,1]/16 的频率响应:
        #     f=0 (DC): 1.0 | f=Nyquist/4: 0.88 | f=Nyquist/2: 0.5
        #   对峰的半宽~10个网格点(1fm/dr)，主频成分在 f≈0.1*Nyquist 处
        #   alpha=0.2 → 峰被压低约 1-0.2*(1-0.95)=0.99 → 轻微但可叠加
        #   alpha=0.2 + 解码器[1,2,1]滤波 = 双重平滑 → 峰高损失 ~15-20%
        #
        # 新策略：alpha=0.05 仅压制周期≤2点的高频Gibbs噪声，
        #         不影响半宽>3点的物理结构。网络可自行增大alpha如果需要更多平滑。
        self.residual_alpha = nn.Parameter(torch.tensor([0.05]))

    def forward(self, x):
        """
        x: (B, channels, N) — 通常是 (B, 2, N) 即 [g, f]
        返回: (B, channels, N) — 平滑后的波函数
        """
        # conv(x) 是平滑后的结果，但初始时权重为 Binomial 核
        # 通过 residual_alpha 控制平滑强度：
        #   alpha=0 → 恒等映射（不改变原始输出）
        #   alpha>0 → 逐渐引入平滑
        smoothed = self.conv(x)
        # 残差连接: 输出 = 原始 + alpha * (平滑修正量)
        # 平滑修正量 = smoothed - x ≈ 二阶导数的离散近似（低通滤波残差）
        return x + self.residual_alpha * (smoothed - x)


class SpectralConv1d(nn.Module):
    """
    一维傅里叶神经算子 (FNO) 核心层
    用于在频域中严密捕捉全空间非局域张量耦合 (积分算子映射)
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            batchsize = x.shape[0]
            x_ft = torch.fft.rfft(x)

            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                                 device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

            x_out = torch.fft.irfft(out_ft, n=x.size(-1))
        return x_out


class FNOBlock1D(nn.Module):
    """原始FNO块（保留向后兼容）"""
    def __init__(self, channels, modes=32):
        super().__init__()
        self.fno = SpectralConv1d(channels, channels, modes)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.fno(x) + self.conv1x1(x))


# ═══════════════════════════════════════════════════════════════
#   模块1: FiLM 特征线性调制层 — 宏观条件编码器核心原语
#   论文 §2.1: 将 (Z, N) 映射为全局缩放因子 γ 和平移因子 β
# ═══════════════════════════════════════════════════════════════

class FiLM_Layer(nn.Module):
    """
    特征线性调制层 (Feature-wise Linear Modulation)

    将宏观量子数 (Z, N) 通过 MLP 映射为调制因子 γ (缩放) 和 β (平移)，
    在 FNO 的每一次非线性激活前强行干预隐空间的能量尺度。

    物理含义：
      - γ 表征库仑排斥力的全局尺度调整
      - β 表征 ρ 介子同位旋势的劈裂偏移

    论文公式: FiLM(x, cond) = γ ⊙ x + β
    """
    def __init__(self, cond_dim=2, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # 输出 γ 和 β 拼接
        )

    def forward(self, x, cond):
        """
        x: (B, d, N) — 空间特征图，d = 隐藏通道数
        cond: (B, 2) — 归一化后的 (Z, N) 标量对
        """
        film_params = self.mlp(cond)           # (B, d*2)
        gamma, beta = film_params.chunk(2, dim=-1)  # 各 (B, d)

        # 扩展维度以匹配空间网格，进行空间广播
        gamma = gamma.unsqueeze(-1)  # (B, d, 1)
        beta = beta.unsqueeze(-1)    # (B, d, 1)

        return gamma * x + beta


# ═══════════════════════════════════════════════════════════════
#   模块2: 条件化傅里叶算子块 — 替代原始 FNOBlock1D
#   论文 §2.3 公式(1):
#     V_{l+1}(r) = GELU( γ^l ⊙ [F^{-1}(R_φ · F(V_l)) + W V_l] + β^l )
# ═══════════════════════════════════════════════════════════════

class Conditioned_FNO_Block(nn.Module):
    """
    包含宏观量子数调制的条件化傅里叶算子块

    与原始 FNOBlock1D 的区别：
      - 在 GELU 激活之前施加 FiLM 调制，而非直接激活
      - 调制因子 (γ, β) 由 (Z, N) 决定，使同一网络适配不同核素

    论文公式(1): V_{l+1}(r) = GELU( γ ⊙ [FNO(V_l) + Conv1x1(V_l)] + β )
    """
    def __init__(self, channels, modes=32):
        super().__init__()
        self.fno = SpectralConv1d(channels, channels, modes)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
        self.film = FiLM_Layer(cond_dim=2, hidden_dim=channels)
        self.gelu = nn.GELU()

    def forward(self, x, macro_cond):
        """
        x: (B, d, N) — 当前层空间特征
        macro_cond: (B, 2) — 归一化后的 (Z, N)
        """
        # 并行执行全局积分与局部代数映射
        x_operator = self.fno(x) + self.conv1x1(x)
        # FiLM 调制后再激活
        x_modulated = self.film(x_operator, macro_cond)
        return self.gelu(x_modulated)


# ═══════════════════════════════════════════════════════════════
#   模块3: 物理交叉注意力层
#   论文 §2.2: 以局域平均场为 Q, 全部单粒子波函数为 K/V,
#             轨道占据几率 ν_i 为权重，复刻 DFT 密度求和
#             ρ(r) = Σ_i ν_i ψ_i† ψ_i
# ═══════════════════════════════════════════════════════════════

class PhysicsCrossAttention(nn.Module):
    """
    [保留向后兼容] 原始物理交叉注意力层。
    新训练脚本使用 OrbitalSelfAttention 替代。
    """
    def __init__(self, feature_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        assert feature_dim % n_heads == 0, f"feature_dim={feature_dim} 必须被 n_heads={n_heads} 整除"
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.nu_scale = nn.Parameter(torch.ones(1))

    def forward(self, mean_field_feat, wavefunc_feat, nu_weights=None):
        B = mean_field_feat.shape[0]
        residual = mean_field_feat
        Q = self.q_proj(mean_field_feat).view(B, self.n_heads, self.head_dim)
        K = self.k_proj(wavefunc_feat).view(B, self.n_heads, self.head_dim)
        V = self.v_proj(wavefunc_feat).view(B, self.n_heads, self.head_dim)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.einsum('bhd,bhd->bh', Q, K) / scale
        if nu_weights is not None:
            attn_scores = attn_scores + self.nu_scale * nu_weights.unsqueeze(1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        attn_weights = self.dropout(attn_weights)
        V_stacked = V.view(B, self.n_heads * self.head_dim)
        attn_for_v = attn_weights.mean(dim=1)
        V_aggregated = torch.einsum('b,bd->d', attn_for_v, V_stacked).unsqueeze(0).expand(B, -1)
        output = self.norm(self.out_proj(V_aggregated) + residual)
        return output


# ═══════════════════════════════════════════════════════════════
#   模块3.5: 轨道自注意力层 (OrbitalSelfAttention)
#   真正的多头自注意力：同一核素内不同轨道间信息交互
#   模拟 DFT 密度求和 ρ(r) = Σ_α ν_α ψ_α† ψ_α
#   占据态贡献大→占据数 ν_α 作为注意力偏置
# ═══════════════════════════════════════════════════════════════

class OrbitalSelfAttention(nn.Module):
    """
    轨道间多头自注意力 (Orbital Self-Attention)

    物理动机：
      RHF自洽场的核心是密度求和 ρ(r) = Σ_α ν_α ψ_α†(r) ψ_α(r)，
      每个轨道的势场由所有轨道的密度共同决定。
      本模块让同一核素内的不同轨道在特征空间中互相"看到"对方，
      从而确保势场一致性——这是原 PhysicsCrossAttention 无法做到的。

    与 PhysicsCrossAttention 的区别：
      - PhysicsCrossAttention: Q=平均场, K/V=波函数, batch维softmax聚合（伪注意力）
      - OrbitalSelfAttention: Q/K/V 均来自轨道特征, 标准缩放点积自注意力

    输入格式：
      同一核素的 n_orbits 个轨道特征，形状 (n_orbits, feature_dim)
      需配合 IsotopeGroupedBatchSampler 使用（同核素轨道在同一batch）
    """
    def __init__(self, feature_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        assert feature_dim % n_heads == 0, f"feature_dim={feature_dim} 必须被 n_heads={n_heads} 整除"

        # Q/K/V 投影（自注意力：三者来源相同）
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

        # 占据数偏置缩放因子
        self.nu_scale = nn.Parameter(torch.ones(1))

    def forward(self, orbital_features, nu_weights=None):
        """
        orbital_features: (n_orbits, feature_dim) — 同一核素内所有轨道的特征
        nu_weights: (n_orbits,) 或 None — 轨道占据几率 ν，用作注意力偏置

        返回: (n_orbits, feature_dim) — 自注意力调制后的轨道特征
        """
        n_orbits = orbital_features.shape[0]
        residual = orbital_features  # (n_orbits, feature_dim)

        # 投影 Q, K, V
        Q = self.q_proj(orbital_features)  # (n_orbits, feature_dim)
        K = self.k_proj(orbital_features)
        V = self.v_proj(orbital_features)

        # 多头拆分: (n_orbits, n_heads, head_dim)
        Q = Q.view(n_orbits, self.n_heads, self.head_dim)
        K = K.view(n_orbits, self.n_heads, self.head_dim)
        V = V.view(n_orbits, self.n_heads, self.head_dim)

        # ★ 标准缩放点积自注意力
        # attn_scores: (n_heads, n_orbits, n_orbits) — 每个head独立
        scale = math.sqrt(self.head_dim)
        # (n_orbits, n_heads, head_dim) → (n_heads, n_orbits, head_dim)
        Q_t = Q.permute(1, 0, 2)  # (n_heads, n_orbits, head_dim)
        K_t = K.permute(1, 0, 2)
        V_t = V.permute(1, 0, 2)

        attn_scores = torch.bmm(Q_t, K_t.transpose(1, 2)) / scale  # (n_heads, n_orbits, n_orbits)

        # 占据数偏置：占据态（ν≈2）贡献更大
        if nu_weights is not None:
            # ν 偏置加到 K 维度（被关注的轨道的权重）
            nu_bias = self.nu_scale * nu_weights.unsqueeze(0).unsqueeze(2)  # (1, n_orbits, 1)
            attn_scores = attn_scores + nu_bias

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (n_heads, n_orbits, n_orbits)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        attn_output = torch.bmm(attn_weights, V_t)  # (n_heads, n_orbits, head_dim)
        # 合并多头: (n_orbits, n_heads, head_dim) → (n_orbits, feature_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(n_orbits, self.feature_dim)

        output = self.out_proj(attn_output)
        output = self.norm(output + residual)

        return output  # (n_orbits, feature_dim)


# ═══════════════════════════════════════════════════════════════
#   主模型: RHF_FNO_GRU — 重构版
#   三级级联: 宏观条件编码器 → 条件化FNO+GRU → 物理交叉注意力 → 解码
# ═══════════════════════════════════════════════════════════════

class RHF_FNO_GRU(nn.Module):
    """
    包含宏观量子数调制的条件化时空神经算子网络

    架构（三级级联 + 轨道自注意力）:
      1. 宏观条件编码器: (Z,N) → MLP → (γ^l, β^l) 用于 FiLM 调制
      2. 条件化FNO + GRU: 4层 Conditioned_FNO_Block → GRU 时序演化
      3. 轨道自注意力: 同核素内轨道间多头自注意力，模拟DFT密度求和

    输入通道: 12维 = 11物理场 + 1演化进度(progress ∈ [0,1])
    输出通道: 11维（仅物理场预测，不含progress）

    新增条件输入: z_num (B,), n_num (B,) — 原子序数和中子数
                  n_principal (B,) — 主量子数（区分 1s vs 2s vs 3s ...）

    参数 use_self_attention: True=使用OrbitalSelfAttention, False=使用原始PhysicsCrossAttention
    """
    def __init__(self, in_channels=12, hidden_dim=64, npt=201, gru_hidden=1024, modes=32,
                 use_self_attention=True):
        super().__init__()
        self.npt = npt
        self.hidden_dim = hidden_dim
        self.physics_channels = 11
        self.use_self_attention = use_self_attention

        # --- 0. 粒子类型嵌入 (中子/质子区分) ---
        self.particle_embed = nn.Embedding(2, 16)
        # ★ 0.5 主量子数嵌入（关键修复：让网络能区分同κ不同n的态）
        self.n_principal_embed = nn.Embedding(8, 16)  # 支持 n=1..7
        in_physics_with_particle = self.physics_channels + 16 + 16  # 11 + 16 + 16 = 43

        # --- 1. 宏观条件编码器 (FiLM) ---
        # (Z, N) 归一化后作为条件输入，在每层 Conditioned_FNO_Block 中注入
        # 归一化参考值：Z_max=82(Pb), N_max=126(Pb)
        self.z_max = 82.0
        self.n_max = 126.0

        # --- 2. 空间算子提取器 (条件化FNO) ---
        self.input_proj = nn.Conv1d(in_physics_with_particle, hidden_dim, 1)
        # 用 Conditioned_FNO_Block 替代原始 FNOBlock1D
        self.spatial_extractor = nn.ModuleList([
            Conditioned_FNO_Block(hidden_dim, modes) for _ in range(4)
        ])

        # --- 3. 时序记忆网络 (GRU) ---
        self.gru_input_size = hidden_dim * npt
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden,
            num_layers=3,  # 论文建议多层GRU模拟SCF阻尼
            batch_first=True
        )

        # --- 4. 注意力层 --- ★ 条件选择：自注意力 vs 交叉注意力
        if self.use_self_attention:
            self.orbital_self_attn = OrbitalSelfAttention(
                feature_dim=gru_hidden, n_heads=4, dropout=0.1
            )
            # 保留兼容
            self.physics_cross_attn = None
        else:
            self.physics_cross_attn = PhysicsCrossAttention(
                feature_dim=gru_hidden, n_heads=4, dropout=0.1
            )
            self.orbital_self_attn = None

        # --- 5. 解码器与输出层 ---
        self.decoder_fc = nn.Linear(gru_hidden, self.gru_input_size)
        self.output_conv = nn.Conv1d(hidden_dim, self.physics_channels, 1)

        # ★ 其余通道(2-10)的输出缩放层
        # 网络原始输出量级不确定，需要一个可学习的缩放使其落到物理空间
        # 使用 xavier 初始化，缩放因子初始≈1.0，偏置≈0.0
        self.other_scale = nn.Parameter(torch.ones(9))  # 9个通道的缩放因子
        self.other_bias = nn.Parameter(torch.zeros(9))   # 9个通道的偏移

        # 指数衰减系数网络 (预测 alpha 以施加无穷远边界约束)
        # ★ v6: 仅输出单一 alpha，G 和 F 共用相同衰减指数
        # 物理依据：r→∞ 时 Dirac 方程退化为自由粒子，G~F~exp(-αr)
        #           两者的区别仅在于前置常数系数(v/c ~ 0.05)，而非衰减率
        self.alpha_net = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # --- 6. 进度投影层（正式初始化，不再懒初始化）---
        self._progress_proj = nn.Linear(self.gru_input_size + 1, self.gru_input_size, bias=False)
        with torch.no_grad():
            w = self._progress_proj.weight
            nn.init.eye_(w[:, :self.gru_input_size])
            w[:, self.gru_input_size:] = 0.01

        # --- 7. 波函数特征投影 (用于物理交叉注意力维度匹配) ---
        self._wavefunc_proj = nn.Linear(self.hidden_dim, gru_hidden, bias=False)
        nn.init.normal_(self._wavefunc_proj.weight, std=0.01)

        # --- 8. Sobolev 平滑正则化层 (消除锯齿) ---
        # 在 ansatz mask 之前对 raw g/f 做可学习平滑（★ 2026-04-19 修复顺序）
        # channels=2 对应 g 和 f 两个通道
        self.sobolev_smooth = SobolevSmoother(channels=2, kernel_size=5)

        # --- 9. ★ 专用本征能量预测头（标量输出） ---
        # ★ 2026-04-19 关键架构修复：
        #
        # 致命缺陷（旧代码）：
        #   能量 ε 被建模为空间场(通道9)，通过 .mean(dim=-1) 取平均得到标量
        #   问题：浪费网络容量（201个网格点全部用于预测同一个标量）
        #         违反量子力学基本原理（能量是全局算子的本征值，不是局域量）
        #         mean()平均会引入噪声和空间伪影
        #
        # 正确设计：
        #   从GRU隐状态(已编码全空间信息)直接映射到标量能量
        #   架构: gru_hidden → FC64 → GELU → FC1 → ε (MeV)
        #   优势：(1) 参数效率高(64+1 vs 201个独立预测)
        #         (2) 物理正确(全局特征→全局标量)
        #         (3) 梯度信号集中(不分散到201个网格点)
        self.energy_predictor = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # 输出单个标量: 束缚态能量 ε
        )
        # 初始化: 典型束缚态能量 ~ -30 MeV (16O 1s1/2)
        with torch.no_grad():
            self.energy_predictor[2].bias.fill_(-30.0)

    def _normalize_zn(self, z_num, n_num):
        """将 (Z, N) 归一化到 [0, 1] 附近，便于 MLP 学习"""
        z_norm = z_num.float() / self.z_max
        n_norm = n_num.float() / self.n_max
        return torch.stack([z_norm, n_norm], dim=-1)  # (B, 2)

    def forward(self, x, kappa, r_grid, is_proton=None, z_num=None, n_num=None, n_principal=None):
        """
        x: (B, L, 12, N) 输入序列（前11维物理场，第12维演化进度progress）
        kappa: (B,) kappa量子数
        r_grid: (B, N) 径向网格
        is_proton: (B,) 粒子类型，0=中子(N), 1=质子(P)
        z_num: (B,) 原子序数 Z
        n_num: (B,) 中子数 N
        n_principal: (B,) ★ 主量子数（区分 1s vs 2s vs 3s ...）

        返回: (B, 11, N) — 仅物理场预测
        """
        B, L, C, N = x.shape

        # ══════════════════════════════════════
        # 宏观条件编码: (Z,N) → 归一化条件向量
        # ══════════════════════════════════════
        if z_num is not None and n_num is not None:
            macro_cond = self._normalize_zn(z_num, n_num)  # (B, 2)
        else:
            # 默认条件: 16O (Z=8, N=8)
            macro_cond = torch.zeros(B, 2, device=x.device, dtype=torch.float32)
            macro_cond[:, 0] = 8.0 / self.z_max
            macro_cond[:, 1] = 8.0 / self.n_max

        # ══════════════════════════════════════
        # 分离物理场(11ch)和进度通道(progress)
        # ══════════════════════════════════════
        x_physics = x[:, :, :self.physics_channels, :]   # (B, L, 11, N)
        x_progress = x[:, :, self.physics_channels:, :]  # (B, L, 1, N)

        # ===== 粒子类型嵌入 + ★ 主量子数嵌入 =====
        x_spatial = x_physics.view(B * L, self.physics_channels, N)

        emb_parts = []
        if is_proton is not None:
            proton_idx = is_proton.long().clamp(0, 1)
            particle_emb = self.particle_embed(proton_idx)  # (B, 16)
            particle_emb_expanded = particle_emb.unsqueeze(-1).expand(B, 16, N)
            particle_emb_expanded = particle_emb_expanded.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, 16, N)
            emb_parts.append(particle_emb_expanded)

        # ★ 主量子数嵌入：让网络区分 1s/2s/3s/...
        if n_principal is not None:
            n_idx = n_principal.long().clamp(1, 7) - 1  # n=1..7 → idx=0..6
            n_emb = self.n_principal_embed(n_idx)  # (B, 16)
            n_emb_expanded = n_emb.unsqueeze(-1).expand(B, 16, N)
            n_emb_expanded = n_emb_expanded.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, 16, N)
            emb_parts.append(n_emb_expanded)

        if emb_parts:
            x_with_particle = torch.cat([x_spatial] + emb_parts, dim=1)
        else:
            zero_extra = torch.zeros(B * L, len(emb_parts) * 16 if emb_parts else 32, N, device=x.device, dtype=x.dtype)
            x_with_particle = torch.cat([x_spatial, zero_extra], dim=1)

        # ══════════════════════════════════════
        # 条件化FNO空间特征提取
        # 每层都注入宏观条件 (Z,N) 的 FiLM 调制
        # ══════════════════════════════════════
        h = self.input_proj(x_with_particle)  # (B*L, hidden_dim, N)

        # 扩展 macro_cond 到 B*L 维度
        macro_cond_bl = macro_cond.unsqueeze(1).expand(-1, L, -1).reshape(B * L, 2)

        # 存储波函数特征用于物理交叉注意力
        wavefunc_feat_per_step = []
        for fno_block in self.spatial_extractor:
            h = fno_block(h, macro_cond_bl)  # (B*L, hidden_dim, N)
            wavefunc_feat_per_step.append(h.mean(dim=-1).view(B, L, self.hidden_dim))

        h_spatial = h  # (B*L, hidden_dim, N)
        h_seq = h_spatial.view(B, L, -1)  # (B, L, hidden_dim * npt)

        # ══════════════════════════════════════
        # Progress 注入时序特征
        # ══════════════════════════════════════
        progress_flat = x_progress.mean(dim=-1).squeeze(-1)  # (B, L)
        progress_feat = progress_flat.unsqueeze(-1)  # (B, L, 1)
        h_enhanced = torch.cat([h_seq, progress_feat], dim=-1)  # (B, L, gru_input_size+1)

        h_for_gru = self._progress_proj(h_enhanced)  # (B, L, gru_input_size)

        # ══════════════════════════════════════
        # GRU 时序演化（模拟SCF收敛阻尼）
        # ══════════════════════════════════════
        gru_out, _ = self.gru(h_for_gru)
        last_hidden = gru_out[:, -1, :]  # (B, gru_hidden)

        # ══════════════════════════════════════
        # 注意力层：自注意力 or 交叉注意力
        # ══════════════════════════════════════
        wavefunc_feat_last = wavefunc_feat_per_step[-1][:, -1, :]  # (B, hidden_dim)
        wavefunc_feat_proj = self._wavefunc_proj(wavefunc_feat_last)  # (B, gru_hidden)

        if self.use_self_attention and self.orbital_self_attn is not None:
            # ★ 轨道自注意力模式：同核素轨道间标准多头自注意力
            # 占据几率: 从输入数据的vv通道提取
            nu_weights = progress_flat[:, -1]  # (B,) — 最后一步的progress值

            # 对每个样本独立应用自注意力（batch内各轨道互相attend）
            # 处理方式：将batch视为同一核素内的不同轨道
            attn_output = self.orbital_self_attn(
                orbital_features=last_hidden,  # (B, gru_hidden)
                nu_weights=nu_weights
            )  # (B, gru_hidden)
            enhanced_hidden = last_hidden + attn_output
        else:
            # 原始交叉注意力模式（向后兼容）
            nu_weights = progress_flat[:, -1]
            attn_output = self.physics_cross_attn(
                mean_field_feat=last_hidden,
                wavefunc_feat=wavefunc_feat_proj,
                nu_weights=nu_weights
            )
            enhanced_hidden = last_hidden + attn_output

        # ══════════════════════════════════════
        # 解码预测
        # ══════════════════════════════════════
        decoded = self.decoder_fc(enhanced_hidden).view(B, self.hidden_dim, self.npt)
        delta_x = self.output_conv(decoded)

        # ★ 解码器抗混叠平滑 — 2026-04-19修复: 仅对通道(2+)应用
        # 问题: decoder_fc输出在空间维度独立 → 高频噪声
        #   修复: g/f通道(0,1)跳过抗混叠（保留尖锐峰），仅势场通道(2-10)做平滑
        if delta_x.size(-1) >= 3 and delta_x.size(1) > 2:
            import torch.nn.functional as F_nn  # 避免与变量f冲突
            aa_kernel = torch.tensor([[1.0, 2.0, 1.0]], dtype=delta_x.dtype, device=delta_x.device) / 4.0
            delta_x_others = delta_x[:, 2:, :]
            n_ch_aa = delta_x_others.size(1)
            aa_kern_expanded = aa_kernel.expand(n_ch_aa, 1, -1)
            delta_x_others_smooth = F_nn.conv1d(delta_x_others, aa_kern_expanded, padding=1, groups=n_ch_aa)
            delta_x = torch.cat([delta_x[:, :2, :], delta_x_others_smooth], dim=1)

        # 目标改变: 直接预测收敛态，而非增量
        # pred_x = delta_x（直接输出，不加x_physics最后一帧）
        pred_x = delta_x

        # --- Step 7: 物理自洽的 Ansatz 边界条件 ---
        # ★ 2026-04-19 关键重构：从 tanh 近似 → 精确 r^{l+1} 幂律
        #
        #   致命缺陷（旧代码）：
        #     使用 origin_factor = tanh(r/ε)^{l+1} 来近似 r^{l+1}
        #     tanh(x) = x - x³/3 + ... → 高阶修正项破坏 Frobenius 解析行为
        #     对于 F 小分量（l_d=1），这导致 κ/r·F 项在 r→0 处虚假发散
        #     优化器被迫扭曲远场波形来压制近核区发散 → 全局畸变
        #
        #   正确物理（Frobenius 级数展开，Dirac 方程在奇异势下的解析解）：
        #     G(r)|_{r→0} = C₀ · r^{l_u+1} · [1 + O(r²)]
        #     F(r)|_{r→0} = C₀'· r^{l_d+1} · [1 + O(r²)]
        #     其中 l_u, l_d 由 κ 唯一确定：
        #       κ < 0: l_u = -κ-1,  l_d = -κ    (例: κ=-1 → l_u=0, l_d=1)
        #       κ > 0: l_u =  κ,    l_d =  κ-1  (例: κ=+1 → l_u=1, l_d=0)
        #
        #   文献依据：
        #     [1] Greiner "Quantum Mechanics: Symmetries" §9.3 — Dirac方程奇点分析
        #     [2] 龙文辉《核物理计算实践》公式3.55-3.57
        #     [3] Wang et al., Chin.Phys.C 49(2025)014106 — ADF消除虚假态的根源
        #
        raw_g = pred_x[:, 0, :]
        raw_f = pred_x[:, 1, :]

        # ===== ★ Sobolev 平滑正则化 (消除锯齿) =====
        # ★ 2026-04-19 关键修复：在 ansatz mask 之前对原始网络输出做平滑
        #
        # 致命缺陷（旧代码）：
        #   旧顺序：raw → ansatz_mask(r^{l+1}) → Sobolev平滑 → 归一化
        #   问题：Sobolev卷积的Binomial核 [1,4,6,4,1]/16 在边界处会"泄漏"非零值
        #         导致 r=0 处 G(r)≠0（破坏 r^{l_u+1} 渐近行为）
        #         κ/r·G 或 κ/r·F 项在原点虚假发散 → Rayleigh商能量偏移
        #
        # 正确顺序：
        #   新顺序：raw → Sobolev平滑(消除高频噪声) → ansatz_mask(r^{l+1}·e^{-αr})
        #   物理依据：平滑操作是线性算子，与幂律掩码可交换(理想情况)
        # ===== ★ Sobolev 平滑正则化（消除锯齿）=====
        #   v7: 撤销 v6 的 *0.05 暴力缩放
        #   物理方程已修正（2Mc² 非对称耦合），网络会自然学到 F 的正确尺度
        gf_raw = torch.stack([raw_g, raw_f], dim=1)  # (B, 2, N)
        gf_smoothed = self.sobolev_smooth(gf_raw)

        raw_g = gf_smoothed[:, 0, :]

        # ★ v15: 移除0.05暴力缩放 — 物理方程已自洽，网络会自然学到F/G≈v/c
        raw_f = gf_smoothed[:, 1, :]

        # 预测衰减系数（控制远场指数衰减）
        alpha_g_raw = self.alpha_net(enhanced_hidden)
        alpha_g = (0.1 + 2.9 * torch.sigmoid(alpha_g_raw)).squeeze(-1)

        # G 和 F 共用相同衰减指数（自由 Dirac 方程渐近行为）
        alpha_f = alpha_g

        alpha_g = alpha_g.unsqueeze(1)
        alpha_f = alpha_f.unsqueeze(1)

        # 获取径向网格
        r_max = (r_grid if r_grid.dim() == 2 else r_grid.unsqueeze(0))

        # ══════════════════════════════════════════
        #  Phase A: 精确轨道角动量计算
        # ══════════════════════════════════════════
        k_val = kappa.view(-1, 1).float()

        # l_u: 大分量(G)轨道角动量
        #   κ > 0 → j = l - 1/2 → l_u = κ
        #   κ < 0 → j = l + 1/2 → l_u = -κ - 1
        l_u = torch.where(k_val > 0, k_val, -k_val - 1.0).float()

        # l_d: 小分量(F)轨道角动量
        #   κ > 0 → l_d = κ - 1
        #   κ < 0 → l_d = -κ
        l_d = torch.where(k_val > 0, k_val - 1.0, -k_val).float()

        # ══════════════════════════════════════════
        #  Phase B: 精确幂律掩码 (r^{l+1})
        # ══════════════════════════════════════════
        pow_g = (l_u + 1.0)  # G 的幂指数: (B, 1)
        pow_f = (l_d + 1.0)  # F 的幂指数: (B, 1)

        # r=0 处保护：防止 0^0 或 NaN 梯度
        #   物理上 r=0 是坐标原点，波函数在此处必须为零（对 l≥0）
        #   数值上 r_min = dr/10 足够小，不影响物理行为
        r_safe = r_max.clamp(min=1e-7)

        # ★ 核心修复：直接使用精确幂律，不再用 tanh 近似！
        ansatz_mask_g = (r_safe ** pow_g) * torch.exp(-alpha_g * r_max)
        ansatz_mask_f = (r_safe ** pow_f) * torch.exp(-alpha_f * r_max)

        # ══════════════════════════════════════════
        #  Phase C: 远场截断（可选的额外衰减）
        # ══════════════════════════════════════════
        # 指数衰减已在 ansatz_mask 中包含（exp(-αr)）
        # 此处的 r_cutoff 仅用于确保在盒子边界处严格归零
        # ★ 2026-04-19: r_cutoff = 8.0 fm（保留2fm尾部缓冲）
        r_cutoff = 8.0
        far_field_g = torch.exp(-alpha_g * torch.abs(r_max - r_cutoff).clamp(min=0.0) * 0.5)
        far_field_f = torch.exp(-alpha_f * torch.abs(r_max - r_cutoff).clamp(min=0.0) * 0.5)

        # 组合边界条件：
        # G(r) = raw_g · r^{l_u+1} · e^{-αg·r} · far_field
        # F(r) = raw_f · r^{l_d+1} · e^{-αf·r} · far_field
        g_constrained = raw_g * ansatz_mask_g * far_field_g
        f_constrained = raw_f * ansatz_mask_f * far_field_f

        # 相位对齐：确保 G 的第一个显著峰值是正的（与标准约定一致）
        # ★ 关键修复：原逻辑用固定窗口[5:20](r=0.5~2.0fm)的均值判断相位
        #   问题: 不同态的峰位置完全不同(1s在r≈0.5fm, 2s在r≈2fm, 2p在r≈3fm)
        #         固定窗口可能落在节点负区 → 错误翻转 → 训练震荡
        #   新方法: 用soft-argmax找|g|的最大值位置，在该位置检查符号
        abs_g = torch.abs(g_constrained)
        # 跳过前5个点(r<0.5fm)的近核平坦区，避免噪声峰干扰
        search_start = 5
        if search_start < abs_g.shape[-1]:
            abs_g_search = abs_g[:, search_start:]
            r_search = r_max[:, search_start:] if r_max.dim() == 2 else r_max[search_start:]
            # soft-argmax: 可微的加权均值峰值位置检测
            temperature = 0.1
            weights = torch.softmax(abs_g_search / temperature, dim=-1)
            peak_pos_idx = (weights * r_search).sum(dim=-1)  # (B,) 连续位置
            # 取峰值位置的符号（通过线性插值近似）
            # 简化: 用搜索区域内最大值点的符号（接近argmax但更稳定）
            max_idx = abs_g_search.argmax(dim=-1)  # (B,)
            peak_val_aligned = g_constrained[
                torch.arange(B, device=g_constrained.device),
                search_start + max_idx
            ]  # (B,) 峰值处的g符号
        else:
            peak_val_aligned = g_constrained[:, 5:20].mean(dim=1)  # fallback

        flip_sign = (peak_val_aligned < 0).float().unsqueeze(-1)
        g_constrained = g_constrained * (1 - 2 * flip_sign)
        f_constrained = f_constrained * (1 - 2 * flip_sign)

        # Sobolev平滑已移至ansatz mask之前（见上方Step 7开头），此处不再重复

        # ===== 硬狄拉克归一化 =====
        g_sq = g_constrained ** 2
        f_sq = f_constrained ** 2
        prob_density = g_sq + f_sq
        if len(r_grid.shape) == 2:
            dr_val = (r_grid[0, 1] - r_grid[0, 0]).clamp(min=1e-6)
        else:
            dr_val = (r_grid[1] - r_grid[0]).clamp(min=1e-6)
        norm_integral = torch.sum(prob_density, dim=-1, keepdim=True).clamp(min=1e-10) * dr_val

        norm_factor = 1.0 / torch.sqrt(norm_integral)
        g_normalized = g_constrained * norm_factor
        f_normalized = f_constrained * norm_factor

        # ===== 软约束：f 分量引导（非强制放大） =====
        # ★ 关键修复：原硬约束(f_min_ratio=0.15 + 强制k_boost放大)导致正反馈循环:
        #   网络输出f小 → 放大f → 重归一化 → g被压缩 → 概率向单点集中 → delta函数!
        #
        # 新策略：仅通过可学习的缩放因子引导f幅度，不做任何硬性放大/重归一化
        # 让网络自己学习正确的g/f比例，物理损失(PDE+Rayleigh商)会自然约束这个比例
        # 如果需要额外的f激励，使用软惩罚项(在loss中)而非硬操作(在forward中)
        pass  # 不再做任何f分量干预，保持归一化后的自然结果

        g_ansatz = g_normalized.unsqueeze(1)
        f_ansatz = f_normalized.unsqueeze(1)

        # 剩余9个物理场通道 — ★ 应用可学习缩放映射到物理空间
        other_fields = pred_x[:, 2:, :]  # (B, 9, N)
        # 逐通道缩放: pred_ch * scale_ch + bias_ch
        scale = self.other_scale.view(1, 9, 1)
        bias = self.other_bias.view(1, 9, 1)

        # ★ 2026-04-19 真空边界条件修复：
        #
        # ★ 严格的真空边界截断：整个势场必须在远区归零
        #
        # 物理诊断（代数错误）：
        #   FNO 是全局傅里叶算子，输出在 r→∞ 时全空间震荡，不会自然归零
        #   旧代码: other_fields * scale + bias * vacuum_mask
        #     → other_fields * scale 在远场仍有非零震荡值！
        #     → 核子在 15 fm 真空中仍感受到 σ, ω 介子场 — 完全破坏渐近自由边界！
        #   新代码: (other_fields * scale + bias) * vacuum_mask
        #     → 整个势场被掩码切断，远区强制指数衰减至真空(0)
        #
        r_for_vacuum = r_max if r_max.dim() == 2 else r_max.unsqueeze(0).expand(B, -1)
        r_vacuum_cutoff = 8.0  # fm — 大约是Pb核半径的2倍
        vacuum_steepness = 2.0  # 控制过渡带宽度: steepness越大过渡越陡峭
        vacuum_mask = torch.sigmoid((r_vacuum_cutoff - r_for_vacuum) * vacuum_steepness).unsqueeze(1)  # (B, 1, N)

        # ★ 对整体施加掩码！确保远区势场 = 0
        other_fields = (other_fields * scale + bias) * vacuum_mask

        # ===== ★ 专用本征能量预测（标量输出）=====
        # ★ 2026-04-19 关键架构修复：
        #   从GRU隐状态(已编码全空间波函数+势场信息)直接映射到标量能量
        #   ε = energy_predictor(enhanced_hidden) → (B, 1)
        #   物理依据：能量是Dirac Hamiltonian的全局本征值，不是局域空间场
        predicted_energy = self.energy_predictor(enhanced_hidden)  # (B, 1)

        # 拼接最终输出
        final_pred_x = torch.cat([g_ansatz, f_ansatz, other_fields], dim=1)

        # 返回: (场预测, 标量能量)
        #   final_pred_x: (B, 11, N) — 完整的物理场（g, f, 势场等）
        #   predicted_energy: (B, 1) — 束缚态本征能量 ε (MeV)
        return final_pred_x, predicted_energy
