"""
增强版模型：通道注意力 + 空间注意力
基于原始 Model_Architecture.py，增加可插拔的注意力机制

使用方法：
    from Model_Architecture_Attention import RHF_FNO_GRU_Attn
    model = RHF_FNO_GRU_Attn(in_channels=11, hidden_dim=64, npt=201, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 注意力机制模块（可独立测试）
# =============================================================================

class ChannelAttention(nn.Module):
    """
    通道注意力 (SE-Net 风格)
    自适应调整 11 个物理通道的重要性权重

    物理直觉：
        - g, f 波函数通道 vs 势场通道应该有不同权重
        - 某些核素/状态下某些通道更重要
    """
    def __init__(self, channels, reduction=4):
        """
        channels: 输入通道数
        reduction: 降维比例，越大参数量越少
        """
        super().__init__()
        self.reduction = reduction
        
        # 双路全局池化 (Squeeze)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Excitation: 先压缩再恢复
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            # 注意：这里不使用 Sigmoid，而是在 forward 中用 Sigmoid
        )
        
    def forward(self, x):
        """
        x: (B, C, N)  - Batch, Channels, Spatial_points
        return: (B, C, N) - 注意力加权的特征
        """
        b, c, n = x.size()
        
        # 双路池化得到全局统计
        avg_out = self.avg_pool(x).view(b, c)           # (B, C)
        max_out = self.max_pool(x).view(b, c)           # (B, C)
        
        # 双路 FC 变换
        avg_weight = self.fc(avg_out)                    # (B, C)
        max_weight = self.fc(max_out)                    # (B, C)
        
        # 融合 + Sigmoid 归一化到 [0, 1]
        weight = (avg_weight + max_weight).unsqueeze(-1)  # (B, C, 1)
        weight = torch.sigmoid(weight)
        
        return x * weight


class SpatialAttention(nn.Module):
    """
    空间注意力 (CBAM 风格)
    聚焦物理关键区域：核表面 (r~2-3fm)、渐近区 (r>10fm)

    物理直觉：
        - 核表面是物理最活跃区域
        - 渐近区决定散射/衰变行为
        - r=0 附近有库仑奇异，需要压制
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
    def forward(self, x):
        """
        x: (B, C, N) - Batch, Channels, Spatial_points
        return: (B, C, N) - 空间注意力加权的特征
        """
        # 双路通道压缩
        avg_out = torch.mean(x, dim=1, keepdim=True)      # (B, 1, N)
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B, 1, N)
        
        # 空间卷积生成注意力图
        scale = torch.cat([avg_out, max_out], dim=1)     # (B, 2, N)
        scale = self.conv(scale)                          # (B, 1, N)
        scale = torch.sigmoid(scale)                      # (B, 1, N)
        
        return x * scale


class ChannelSpatialAttention(nn.Module):
    """
    通道-空间联合注意力 (CBAM 完整版)
    先通道注意力，再空间注意力
    """
    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)
        
    def forward(self, x):
        """
        x: (B, C, N)
        1. 先通道注意力：调整通道重要性
        2. 再空间注意力：聚焦空间关键区域
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ProgressiveAttention(nn.Module):
    """
    渐进式注意力：在 FNO 多层中逐步增强注意力
    浅层提取基础特征，深层关注精细结构
    """
    def __init__(self, channels, reduction=4, spatial_kernel=7, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # 每层独立的注意力，浅层降低注意力（保留更多细节）
        # 深层增强注意力（聚焦关键特征）
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                ChannelAttention(channels, reduction=max(2, reduction - i)),
                SpatialAttention(spatial_kernel)
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x_list):
        """
        x_list: FNO 各层的输出列表 [layer0_out, layer1_out, ...]
        return: 注意力增强后的特征列表
        """
        out_list = []
        for i, x in enumerate(x_list):
            attn_out = self.attention_layers[i](x)
            out_list.append(attn_out)
        return out_list


# =============================================================================
# 测试函数
# =============================================================================

def test_attention_modules():
    """独立测试注意力模块"""
    print("=" * 60)
    print("注意力模块单元测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, N = 4, 11, 201  # Batch, Channels, Spatial_points
    
    # 模拟输入
    x = torch.randn(B, C, N).to(device)
    print(f"\n输入张量: {x.shape}")
    
    # === 测试通道注意力 ===
    print("\n--- ChannelAttention ---")
    ch_attn = ChannelAttention(channels=C, reduction=4).to(device)
    out_ch = ch_attn(x)
    print(f"输出张量: {out_ch.shape}")
    
    # 检查输出范围
    print(f"输入范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"输出范围: [{out_ch.min():.3f}, {out_ch.max():.3f}]")
    
    # 检查通道权重分布
    with torch.no_grad():
        avg_pool = nn.AdaptiveAvgPool1d(1)(x)
        max_pool = nn.AdaptiveMaxPool1d(1)(x)
        print(f"g/f 通道 (0,1) 池化均值: {avg_pool[0, [0,1], 0].cpu().numpy()}")
    
    # === 测试空间注意力 ===
    print("\n--- SpatialAttention ---")
    sp_attn = SpatialAttention(kernel_size=7).to(device)
    out_sp = sp_attn(x)
    print(f"输出张量: {out_sp.shape}")
    
    # === 测试联合注意力 ===
    print("\n--- ChannelSpatialAttention ---")
    cs_attn = ChannelSpatialAttention(channels=C, reduction=4).to(device)
    out_cs = cs_attn(x)
    print(f"输出张量: {out_cs.shape}")
    
    # === 测试梯度流 ===
    print("\n--- 梯度流测试 ---")
    loss = out_cs.sum()
    loss.backward()
    
    ch_grad_norm = ch_attn.fc.fc[0].weight.grad.norm().item() if ch_attn.fc.fc[0].weight.grad is not None else 0
    sp_grad_norm = sp_attn.conv.weight.grad.norm().item() if sp_attn.conv.weight.grad is not None else 0
    
    print(f"通道注意力 FC 梯度范数: {ch_grad_norm:.6f}")
    print(f"空间注意力 Conv 梯度范数: {sp_grad_norm:.6f}")
    print(f"梯度流正常: {ch_grad_norm > 0 and sp_grad_norm > 0}")
    
    # === 参数量统计 ===
    print("\n--- 参数量统计 ---")
    total_params = sum(p.numel() for p in ch_attn.parameters())
    print(f"ChannelAttention 参数量: {total_params:,}")
    
    total_params = sum(p.numel() for p in sp_attn.parameters())
    print(f"SpatialAttention 参数量: {total_params:,}")
    
    total_params = sum(p.numel() for p in cs_attn.parameters())
    print(f"ChannelSpatialAttention 参数量: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return ch_attn, sp_attn, cs_attn


if __name__ == "__main__":
    test_attention_modules()
