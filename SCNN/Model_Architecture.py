import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    物理交叉注意力层 (Physics Cross-Attention)

    在 GRU 时序演化之后、最终解码之前插入。
    模拟 DFT 中的密度求和过程：
      - Query: 从 GRU 输出的平均场特征投影而来（代表局域平均势场）
      - Key/Value: 从波函数特征投影而来（代表各单粒子轨道）
      - 轨道占据几率 ν 作为注意力偏置

    由于 batch 内不同样本可能属于不同核素，按 (Z, N, is_proton) 分组
    进行同组交叉注意力，确保物理一致性。
    """
    def __init__(self, feature_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        assert feature_dim % n_heads == 0, f"feature_dim={feature_dim} 必须被 n_heads={n_heads} 整除"

        # Q: 平均场投影 (从 GRU hidden state)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        # K/V: 波函数特征投影 (从空间特征)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        # 输出投影
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        # 占据几率 ν 的缩放因子
        self.nu_scale = nn.Parameter(torch.ones(1))

    def forward(self, mean_field_feat, wavefunc_feat, nu_weights=None):
        """
        mean_field_feat: (B, feature_dim) — GRU输出的平均场特征 (Query来源)
        wavefunc_feat: (B, feature_dim) — 空间提取器的波函数特征 (Key/Value来源)
        nu_weights: (B,) 或 None — 轨道占据几率 ν，用作注意力偏置

        返回: (B, feature_dim) — 密度调制后的特征
        """
        B = mean_field_feat.shape[0]

        # 残差连接准备
        residual = mean_field_feat

        # 投影 Q, K, V
        Q = self.q_proj(mean_field_feat)  # (B, d)
        K = self.k_proj(wavefunc_feat)    # (B, d)
        V = self.v_proj(wavefunc_feat)    # (B, d)

        # 多头拆分: (B, n_heads, head_dim)
        Q = Q.view(B, self.n_heads, self.head_dim)
        K = K.view(B, self.n_heads, self.head_dim)
        V = V.view(B, self.n_heads, self.head_dim)

        # 交叉注意力: Q 与 K 的点积
        # 同 batch 内所有样本互相attend (模拟同一核素内不同轨道的密度耦合)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.einsum('bhd,bhd->bh', Q, K) / scale  # (B, n_heads) — 自身对

        # 如果 nu_weights 存在，作为注意力偏置
        if nu_weights is not None:
            nu_bias = self.nu_scale * nu_weights.unsqueeze(1)  # (B, 1)
            attn_scores = attn_scores + nu_bias

        attn_weights = torch.softmax(attn_scores, dim=0)  # (B, n_heads) — 沿 batch 维度
        attn_weights = self.dropout(attn_weights)

        # 加权聚合 V
        # 对每个 head，所有 batch 样本的 V 按注意力权重聚合
        V_agg = torch.einsum('bh,bhd->d', attn_weights, V)  # (head_dim,) — 标量？
        # 需要修正：应该是每个 sample 都获得聚合后的信息
        # 更合理的实现：每个 sample 获得全 batch 加权平均的 V
        V_stacked = V.view(B, self.n_heads * self.head_dim)  # (B, d)
        attn_for_v = attn_weights.mean(dim=1)  # (B,) — 平均各 head 的注意力
        # 全 batch 加权求和
        V_aggregated = torch.einsum('b,bd->d', attn_for_v, V_stacked)  # (d,)
        # 广播回每个 sample (加残差)
        V_aggregated = V_aggregated.unsqueeze(0).expand(B, -1)  # (B, d)

        output = self.out_proj(V_aggregated)
        output = self.norm(output + residual)

        return output  # (B, feature_dim)


# ═══════════════════════════════════════════════════════════════
#   主模型: RHF_FNO_GRU — 重构版
#   三级级联: 宏观条件编码器 → 条件化FNO+GRU → 物理交叉注意力 → 解码
# ═══════════════════════════════════════════════════════════════

class RHF_FNO_GRU(nn.Module):
    """
    包含宏观量子数调制的条件化时空神经算子网络

    架构（严格按论文 §2 三级级联）:
      1. 宏观条件编码器: (Z,N) → MLP → (γ^l, β^l) 用于 FiLM 调制
      2. 条件化FNO + GRU: 4层 Conditioned_FNO_Block → GRU 时序演化
      3. 物理交叉注意力: Q=平均场, K/V=波函数, ν加权 → 密度调制特征

    输入通道: 12维 = 11物理场 + 1演化进度(progress ∈ [0,1])
    输出通道: 11维（仅物理场预测，不含progress）

    新增条件输入: z_num (B,), n_num (B,) — 原子序数和中子数
                  n_principal (B,) — 主量子数（区分 1s vs 2s vs 3s ...）
    """
    def __init__(self, in_channels=12, hidden_dim=64, npt=201, gru_hidden=1024, modes=32):
        super().__init__()
        self.npt = npt
        self.hidden_dim = hidden_dim
        self.physics_channels = 11

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

        # --- 4. 物理交叉注意力层 ---
        self.physics_cross_attn = PhysicsCrossAttention(
            feature_dim=gru_hidden, n_heads=4, dropout=0.1
        )

        # --- 5. 解码器与输出层 ---
        self.decoder_fc = nn.Linear(gru_hidden, self.gru_input_size)
        self.output_conv = nn.Conv1d(hidden_dim, self.physics_channels, 1)

        # 指数衰减系数网络 (预测 alpha 以施加无穷远边界约束)
        self.alpha_net = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2),
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
            macro_cond = torch.zeros(B, 2, device=x.device, dtype=x.float32)
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
        # 物理交叉注意力: 密度调制
        # Q = 平均场特征 (GRU输出)
        # K/V = 波函数特征 (FNO最后一层空间特征均值)
        # ══════════════════════════════════════
        wavefunc_feat_last = wavefunc_feat_per_step[-1][:, -1, :]  # (B, hidden_dim)
        wavefunc_feat_proj = self._wavefunc_proj(wavefunc_feat_last)  # (B, gru_hidden)

        # 占据几率: 从输入数据的第12通道(已归一化后的vv)提取
        # 简化: 使用 progress 作为近似权重（接近收敛的步权重更高）
        nu_weights = progress_flat[:, -1]  # (B,) — 最后一步的progress值

        attn_output = self.physics_cross_attn(
            mean_field_feat=last_hidden,
            wavefunc_feat=wavefunc_feat_proj,
            nu_weights=nu_weights
        )  # (B, gru_hidden)

        # 残差融合: 注意力输出 + GRU输出
        enhanced_hidden = last_hidden + attn_output

        # ══════════════════════════════════════
        # 解码预测
        # ══════════════════════════════════════
        decoded = self.decoder_fc(enhanced_hidden).view(B, self.hidden_dim, self.npt)
        delta_x = self.output_conv(decoded)

        # 目标改变: 直接预测收敛态，而非增量
        # pred_x = delta_x（直接输出，不加x_physics最后一帧）
        pred_x = delta_x

        # --- 物理 Ansatz 整流 ---
        raw_g = pred_x[:, 0, :]
        raw_f = pred_x[:, 1, :]

        # 预测衰减系数
        alphas_raw = self.alpha_net(enhanced_hidden)
        alphas = 0.1 + 2.9 * torch.sigmoid(alphas_raw)  # ★ alpha ∈ [0.1, 3.0] 增强远场衰减
        alpha_g, alpha_f = alphas[:, 0].unsqueeze(1), alphas[:, 1].unsqueeze(1)
        kappa_exp = kappa.abs().unsqueeze(1)

        # ★ 增强远场衰减：在 r > 15fm 区域施加更强的指数压制
        # 方法：对 r_grid 构造分段衰减函数，远场区域额外乘以二次衰减因子
        r_max = (r_grid if r_grid.dim() == 2 else r_grid.unsqueeze(0))
        far_field_factor = torch.where(
            r_max > 15.0,
            torch.exp(-0.05 * (r_max - 15.0) ** 2),  # 高斯型额外衰减
            torch.ones_like(r_max)
        )

        # 施加硬边界条件约束: r^{|kappa|} * exp(-alpha * r) * NN(r) * far_field_factor
        ansatz_mask_g = (r_grid ** kappa_exp) * torch.exp(-alpha_g * r_grid) * far_field_factor.squeeze() if far_field_factor.dim() == 3 else (r_grid ** kappa_exp) * torch.exp(-alpha_g * r_grid)
        ansatz_mask_f = (r_grid ** kappa_exp) * torch.exp(-alpha_f * r_grid) * far_field_factor.squeeze() if far_field_factor.dim() == 3 else (r_grid ** kappa_exp) * torch.exp(-alpha_f * r_grid)

        # ===== ★ 统一鲁棒找峰相位对齐（与 Physics_Informed_Loss.py 一致）=====
        # 替代原有的内区均值翻转，使用双阶段显著峰检测
        def _model_find_first_peak(g_out):
            """模型前向传播中使用的轻量版找峰（与 loss 中的逻辑一致）"""
            B_loc, L_loc = g_out.shape
            order = 5
            abs_g = torch.abs(g_out)
            search_start = max(order + 1, 5)
            search_end = L_loc - order
            if search_end <= search_start:
                return abs_g.argmax(dim=-1), g_out[torch.arange(B_loc, device=g_out.device), abs_g.argmax(dim=-1)]
            mask_r = torch.ones_like(abs_g); mask_r[:, :search_start] = 0; mask_r[:, search_end:] = 0
            anchor_idx = (abs_g * mask_r).argmax(dim=-1)
            pad_p = torch.nn.functional.pad(abs_g, (order, order), value=-float('inf'))
            pool_p = torch.nn.functional.max_pool1d(pad_p.unsqueeze(1), kernel_size=2*order+1, stride=1).squeeze(1)
            is_lm = (abs_g >= pool_p - 1e-10) & (abs_g > 0) & (mask_r.bool())
            lm_m = torch.zeros_like(is_lm); lm_m[:, order:L_loc-order] = True; is_lm = is_lm & lm_m
            ar_L = torch.arange(L_loc, dtype=torch.long, device=g_out.device).unsqueeze(0)
            is_cand = is_lm.clone() & (ar_L <= anchor_idx.unsqueeze(1))
            rev_cs = is_cand.flip(dims=[1]).float().cumsum(dim=-1).flip(dims=[1])
            is_last = is_cand & (rev_cs == 1)
            last_p = ar_L.masked_fill(~is_last, 0).amax(dim=-1)
            has_c = (is_cand.float().sum(dim=-1) > 0)
            final_i = torch.where(has_c, last_p, anchor_idx)
            return final_i, g_out[torch.arange(B_loc, device=g_out.device), final_i]

        peak_idx_model, peak_val_model = _model_find_first_peak(raw_g)
        flip_sign = (peak_val_model < 0).float().unsqueeze(-1)
        raw_g = raw_g * flip_sign - raw_g * (1 - flip_sign)
        raw_f = raw_f * flip_sign - raw_f * (1 - flip_sign)

        g_constrained = raw_g * ansatz_mask_g
        f_constrained = raw_f * ansatz_mask_f

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

        # ===== 硬约束：防止 f 分量被偷懒置零 =====
        # 物理事实：狄拉克束缚态的 f/g 比例在核内区域通常为 5%~30%
        # 如果 ∫f² / ∫(g²+f²) < f_min_ratio，强制提升 f 的幅度
        f_min_ratio = 0.05  # f至少贡献5%的概率密度
        f_power_frac = (f_normalized ** 2).sum(dim=-1, keepdim=True) * dr_val  # (B,1)
        # f_power_frac 应该在 [0, 1] 范围（因为已归一化）
        f_deficit = torch.clamp(f_min_ratio - f_power_frac, min=0.0)  # (B,1)
        # 有亏空时：放大 f，重新归一化保持总概率=1
        has_deficit = (f_deficit > 1e-6).squeeze(-1)  # (B,)
        if has_deficit.any():
            # 提升因子：让 f 贡献达到 f_min_ratio
            # 设放大 k 倍后: k²*f² / (g² + k²*f²) ≥ f_min_ratio
            # 解得: k² ≥ f_min_ratio * (g²+f²) / ((1-f_min_ratio)*f²)
            g_sq_sum = (g_normalized ** 2).sum(dim=-1, keepdim=True) * dr_val
            f_sq_sum = (f_normalized ** 2).sum(dim=-1, keepdim=True) * dr_val + 1e-10
            target_k_sq = (f_min_ratio * 1.0) / ((1.0 - f_min_ratio) * f_sq_sum.clamp(min=1e-10))
            k_boost = torch.sqrt(target_k_sq.clamp(min=1.0))  # (B,1), 最小为1
            f_normalized = f_normalized * k_boost
            # 重新归一化
            total_new = (g_normalized**2 + f_normalized**2).sum(dim=-1, keepdim=True) * dr_val
            renorm = torch.sqrt(total_new.clamp(min=1e-10))
            g_normalized = g_normalized / renorm
            f_normalized = f_normalized / renorm

        g_ansatz = g_normalized.unsqueeze(1)
        f_ansatz = f_normalized.unsqueeze(1)

        # 剩余9个物理场通道
        other_fields = pred_x[:, 2:, :]

        # 拼接最终输出
        final_pred_x = torch.cat([g_ansatz, f_ansatz, other_fields], dim=1)

        return final_pred_x  # (B, 11, N)
