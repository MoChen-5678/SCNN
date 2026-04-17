"""
增强版模型：基于通道注意力的 RHF 神经算子
RHF_FNO_GRU + ChannelAttention

使用方法：
    from Model_Architecture_Attention import RHF_FNO_GRU_Attn, SpectralConv1d, FNOBlock1D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention_Modules import ChannelAttention, SpatialAttention, ChannelSpatialAttention


# =============================================================================
# 基础模块（与原始 Model_Architecture.py 保持一致）
# =============================================================================

class SpectralConv1d(nn.Module):
    """
    一维傅里叶神经算子 (FNO) 核心层
    用于在频域中严密捕捉全空间非局域张量耦合
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
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        
        x_out = torch.fft.irfft(out_ft, n=x.size(-1))
        return x_out


class FNOBlock1D(nn.Module):
    """FNO 块：频域全局算子 + 空域局部卷积"""
    def __init__(self, channels, modes=32):
        super().__init__()
        self.fno = SpectralConv1d(channels, channels, modes)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.fno(x) + self.conv1x1(x))


class FNOBlock1D_Attn(nn.Module):
    """
    增强版 FNO 块：FNO + 通道注意力
    在频域/空域变换后加入通道注意力，增强特征选择能力
    """
    def __init__(self, channels, modes=32, use_channel_attn=True, use_spatial_attn=False):
        super().__init__()
        self.fno = SpectralConv1d(channels, channels, modes)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
        
        # 注意力模块
        self.channel_attn = ChannelAttention(channels, reduction=4) if use_channel_attn else None
        self.spatial_attn = SpatialAttention(kernel_size=7) if use_spatial_attn else None
        
        self.gelu = nn.GELU()

    def forward(self, x):
        # 1. FNO 频域变换
        fno_out = self.fno(x)
        
        # 2. 局部卷积
        conv_out = self.conv1x1(x)
        
        # 3. 融合
        fused = self.gelu(fno_out + conv_out)
        
        # 4. 通道注意力 (如果启用)
        if self.channel_attn is not None:
            fused = self.channel_attn(fused)
        
        # 5. 空间注意力 (如果启用)
        if self.spatial_attn is not None:
            fused = self.spatial_attn(fused)
        
        return fused


# =============================================================================
# 主模型：带注意力机制的 RHF 神经算子
# =============================================================================

class RHF_FNO_GRU_Attn(nn.Module):
    """
    物理驱动的时空联合神经算子网络 - 增强版

    新增功能：
    - 通道注意力：自适应调整 11 个物理通道的重要性
    - 可选的渐进式注意力：FNO 各层逐步增强注意力

    Args:
        in_channels: 输入通道数 (默认 11)
        hidden_dim: FNO 隐藏维度 (默认 64)
        npt: 空间网格点数 (默认 201)
        gru_hidden: GRU 隐藏维度 (默认 1024)
        modes: 傅里叶模式数 (默认 32)
        use_channel_attn: 是否使用通道注意力 (默认 True)
        use_spatial_attn: 是否使用空间注意力 (默认 False)
        channel_attn_pos: 通道注意力位置 ['input', 'fno', 'both'] (默认 'both')
    """
    def __init__(self, in_channels=11, hidden_dim=64, npt=201, gru_hidden=1024, 
                 modes=32, use_channel_attn=True, use_spatial_attn=False,
                 channel_attn_pos='both'):
        super().__init__()
        self.npt = npt
        self.hidden_dim = hidden_dim
        self.use_channel_attn = use_channel_attn
        self.channel_attn_pos = channel_attn_pos
        
        # === 1. 输入投影 ===
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)
        
        # === 2. 通道注意力 (输入层) ===
        if use_channel_attn and channel_attn_pos in ['input', 'both']:
            self.input_channel_attn = ChannelAttention(hidden_dim, reduction=4)
        else:
            self.input_channel_attn = None
        
        # === 3. 空间算子提取器 (FNO) ===
        if use_channel_attn:
            # 使用增强版 FNO (每层都有注意力)
            self.spatial_extractor = nn.Sequential(*[
                FNOBlock1D_Attn(hidden_dim, modes, 
                               use_channel_attn=True, 
                               use_spatial_attn=use_spatial_attn) 
                for _ in range(4)
            ])
        else:
            self.spatial_extractor = nn.Sequential(*[
                FNOBlock1D(hidden_dim, modes) 
                for _ in range(4)
            ])
        
        # === 4. 通道注意力 (输出层，FNO 之后) ===
        if use_channel_attn and channel_attn_pos in ['fno', 'both']:
            self.output_channel_attn = ChannelAttention(hidden_dim, reduction=4)
        else:
            self.output_channel_attn = None
        
        # === 5. 时序记忆网络 (GRU) ===
        self.gru_input_size = hidden_dim * npt
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True
        )

        # === 6. 解码器与输出层 ===
        self.decoder_fc = nn.Linear(gru_hidden, self.gru_input_size)
        self.output_conv = nn.Conv1d(hidden_dim, in_channels, 1)
        
        # === 7. 指数衰减系数网络 (边界条件) ===
        self.alpha_net = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softplus()
        )
        
        # === 8. Kappa 预测头 ===
        self.kappa_predictor = nn.Sequential(
            nn.Linear(gru_hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # 打印注意力配置
        self._print_attention_config()
        
    def _print_attention_config(self):
        """打印注意力机制配置"""
        attn_cfg = []
        if self.input_channel_attn is not None:
            attn_cfg.append("输入层通道注意力")
        if hasattr(self.spatial_extractor[0], 'channel_attn') and \
           self.spatial_extractor[0].channel_attn is not None:
            attn_cfg.append("FNO层通道注意力(x4)")
        if self.output_channel_attn is not None:
            attn_cfg.append("输出层通道注意力")
        if attn_cfg:
            print(f"[注意力配置] {' + '.join(attn_cfg)}")
        
    def forward(self, x, kappa, r_grid):
        """
        前向传播

        Args:
            x: (B, L, C, N) - Batch, Seq_len, Channels, Spatial_points
            kappa: (B,) - 量子数
            r_grid: (N,) - 径向网格

        Returns:
            pred_x: (B, C, N) - 预测的波函数
            pred_kappa: (B,) - 预测的量子数
        """
        B, L, C, N = x.shape
        
        # Step 1: 展平序列维度
        x_spatial = x.view(B * L, C, N)
        
        # Step 2: 输入投影 + 通道注意力
        h = self.input_proj(x_spatial)
        if self.input_channel_attn is not None:
            h = self.input_channel_attn(h)
        
        # Step 3: FNO 空间特征提取
        h_spatial = self.spatial_extractor(h)
        
        # Step 4: 输出层通道注意力
        if self.output_channel_attn is not None:
            h_spatial = self.output_channel_attn(h_spatial)
        
        # Step 5: GRU 时序演化
        h_seq = h_spatial.view(B, L, -1)
        gru_out, _ = self.gru(h_seq)
        last_hidden = gru_out[:, -1, :]
        
        # Step 6: 解码预测
        decoded = self.decoder_fc(last_hidden).view(B, self.hidden_dim, self.npt)
        delta_x = self.output_conv(decoded)
        pred_x = x[:, -1, :, :] + delta_x
        
        # Step 7: 物理 Ansatz 边界条件
        raw_g = pred_x[:, 0, :]
        raw_f = pred_x[:, 1, :]
        
        alphas = self.alpha_net(last_hidden)
        alpha_g, alpha_f = alphas[:, 0].unsqueeze(1), alphas[:, 1].unsqueeze(1)
        kappa_exp = kappa.abs().unsqueeze(1)
        
        ansatz_mask_g = (r_grid ** kappa_exp) * torch.exp(-alpha_g * r_grid)
        ansatz_mask_f = (r_grid ** kappa_exp) * torch.exp(-alpha_f * r_grid)
        
        g_ansatz = (raw_g * ansatz_mask_g).unsqueeze(1)
        f_ansatz = (raw_f * ansatz_mask_f).unsqueeze(1)
        other_fields = pred_x[:, 2:, :]
        
        final_pred_x = torch.cat([g_ansatz, f_ansatz, other_fields], dim=1)
        
        # Step 8: Kappa 预测
        pred_kappa = self.kappa_predictor(last_hidden).squeeze(-1)
        
        return final_pred_x, pred_kappa


# =============================================================================
# 测试函数
# =============================================================================

def test_attention_model():
    """测试带注意力机制的模型"""
    print("=" * 60)
    print("带注意力机制的 RHF 神经算子测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数
    B, L, C, N = 4, 8, 11, 201
    hidden_dim, gru_hidden, modes = 64, 1024, 32
    
    # 创建模型
    print("\n创建模型...")
    model = RHF_FNO_GRU_Attn(
        in_channels=C,
        hidden_dim=hidden_dim,
        npt=N,
        gru_hidden=gru_hidden,
        modes=modes,
        use_channel_attn=True,
        use_spatial_attn=False,
        channel_attn_pos='both'
    ).to(device)
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试输入
    print("\n测试前向传播...")
    x = torch.randn(B, L, C, N).to(device)
    kappa = torch.randint(-10, 10, (B,)).float().to(device)
    r_grid = torch.linspace(0.001, 20, N).unsqueeze(0).expand(B, -1).to(device)
    
    # 前向传播
    pred_x, pred_kappa = model(x, kappa, r_grid)
    
    print(f"预测波函数: {pred_x.shape}")
    print(f"预测 kappa: {pred_kappa}")
    print(f"真实 kappa: {kappa}")
    
    # 测试损失
    loss_x = F.mse_loss(pred_x, x[:, -1, :, :])
    loss_kappa = F.mse_loss(pred_kappa, kappa.float())
    print(f"\nMSE Loss (x): {loss_x.item():.6f}")
    print(f"MSE Loss (kappa): {loss_kappa.item():.6f}")
    
    # 测试梯度
    print("\n测试反向传播...")
    loss = loss_x + 0.1 * loss_kappa
    loss.backward()
    
    # 检查注意力模块梯度
    has_attn_grad = any('channel_attn' in name for name, _ in [(n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None])
    print(f"注意力模块梯度正常: {has_attn_grad}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_attention_model()
