#!/usr/bin/env python3
"""
v11 震荡系统性诊断：逐项排除所有可能的数值和物理原因
运行方式: python diagnose_v11_oscillation.py
"""

import sys
sys.path.insert(0, 'SCNN')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Physics_Informed_Loss import (
    get_cached_fd_matrix_5padf,
    _apply_fd_matrix,
    calc_physics_residual,
    calc_simplified_residual
)
from Data_Loader import DataLoader as RHFDataLoader

# ============================================================
#  配置（匹配训练参数）
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

loader = RHFDataLoader(data_path='SCNN/data/rhf_16O.npz')
batch = next(iter(torch.utils.data.DataLoader(loader, batch_size=1)))
batch = {k: v.to(device) for k, v in batch.items()}

r_grid = batch['r_grid']
npt = r_grid.shape[-1]
dr = r_grid[0, 1].item() - r_grid[0, 0].item()
kappa = batch['kappa']
n_principal = batch['n_principal']

print(f"npt={npt}, dr={dr:.4f} fm, kappa={kappa.item()}, n_principal={n_principal.item()}")

# ============================================================
#  Test 1: 差分矩阵数值特性详细检查
# ============================================================
print("\n" + "="*70)
print("Test 1: 差分矩阵数值特性")
print("="*70)

D_fwd = get_cached_fd_matrix_5padf(npt, dr, direction='forward', device=device)
D_bwd = get_cached_fd_matrix_5padf(npt, dr, direction='backward', device=device)

# 检查1: 边界镜像对称
print(f"\n[1a] 边界符号翻转:")
for i, name in [(0, "左边界"), (npt-1, "右边界")]:
    s_fwd = D_fwd[i, max(0,i-5):i+5].cpu().numpy()
    s_bwd = D_bwd[i, max(0,i-5):i+5].cpu().numpy()
    sum_abs = np.abs(s_fwd + s_bwd).max()
    print(f"  {name} i={i}: |D_fwd+D_bwd|max = {sum_abs:.2e} {'✓' if sum_abs < 0.01 else '✗'}")

# 检查2: 内部点一致性（应该相同——都是中心差分）
i_start, i_end = 5, npt - 5
internal_diff = (D_fwd[i_start:i_end, :] - D_bwd[i_start:i_end, :]).abs().max()
print(f"\n[1b] 内部点一致性(i=5:{npt-5}): |D_fwd-D_bwd|max = {internal_diff:.2e} {'✓' if internal_diff < 1e-10 else '✗ 应该≈0'}")

# 检查3: 矩阵条件数（病态程度）
print(f"\n[1c] 矩阵条件数:")
cond_fwd = torch.linalg.cond(D_fwd.float()).item()
cond_bwd = torch.linalg.cond(D_bwd.float()).item()
print(f"  cond(D_forward) = {cond_fwd:.2e}")
print(f"  cond(D_backward) = {cond_bwd:.2e}")
if cond_fwd > 1e6:
    print("  ⚠️ Forward矩阵严重病态！差分算子可能数值不稳定")
if cond_bwd > 1e6:
    print("  ⚠️ Backward矩阵严重病态！差分算子可能数值不稳定")

# 检查4: 特征值谱（纯导数算子的谱性质）
eig_fwd = torch.linalg.eigvals(D_fwd.float()).real
eig_bwd = torch.linalg.eigvals(D_bwd.float()).real
print(f"\n[1d] 特征值范围:")
print(f"  D_forward λ ∈ [{eig_fwd.min():.2f}, {eig_fwd.max():.2f}]")
print(f"  D_backward λ ∈ [{eig_bwd.min():.2f}, {eig_bwd.max():.2f}]")
# 对于一阶导数近似，特征值应该在纯虚数附近（反对称算子）
eig_imag_fwd = torch.linalg.eigvals(D_fwd.float()).imag
eig_imag_bwd = torch.linalg.eigvals(D_bwd.float()).imag
print(f"  D_forward Im(λ) ∈ [{eig_imag_fwd.min():.2f}, {eig_imag_fwd.max():.2f}]")
print(f"  D_backward Im(λ) ∈ [{eig_imag_bwd.min():.2f}, {eig_imag_bwd.max():.2f}]")

# ============================================================
#  Test 2: Rayleigh商逐项分解（核心诊断）
# ============================================================
print("\n" + "="*70)
print("Test 2: Rayleigh商逐项分解 — 找出震荡源")
print("="*70)

# 用一个简单的测试波函数来隔离各项贡献
# 模拟类1s波函数: G ~ r*exp(-alpha*r), F ~ 0.01*r^2*exp(-alpha*r)
r = torch.arange(npt, device=device, dtype=torch.float32) * dr
r[0] = 0.001
r = r.unsqueeze(0)

alpha_test = 0.8  # 衰减常数 fm^-1
g_test = r * torch.exp(-alpha_test * r)       # 类s1/2大分量
g_test = g_test / torch.sqrt(torch.sum(g_test**2) * dr)  # 归一化
f_test = 0.03 * r**2 * torch.exp(-alpha_test * r)  # 小分量
f_test = f_test / torch.sqrt(torch.sum(f_test**2).clamp(min=1e-10) * dr)
# 重新归一化整体
norm = torch.sqrt(torch.sum(g_test**2 + f_test**2) * dr)
g_test = g_test / norm
f_test = f_test / norm

print(f"\n测试波函数: <g²>={torch.sum(g_test**2).item()*dr:.4f}, <f²>={torch.sum(f_test**2).item()*dr:.4f}")
print(f"  peak(|g|)={g_test.abs().max():.4f}, peak(|f|)={f_test.abs().max():.4f}")
print(f"  F/G幅度比 ≈ {f_test.abs().max()/g_test.abs().max():.4f} (物理预期~0.04)")

# 计算导数
dg_test = _apply_fd_matrix(g_test.squeeze(), D_fwd).unsqueeze(0)  # (1,N)
df_test = _apply_fd_matrix(f_test.squeeze(), D_bwd).unsqueeze(0)  # (1,N)

# 从batch获取势场
vps = batch['vps']     # (1, N)
vms = batch['vms']     # (1, N)
vtt = batch['vtt']     # (1, N)
XG = batch['XG']       # (1, N)
XF = batch['XF']       # (1, N)
YG = batch['YG']       # (1, N)
YF = batch['YF']       # (1, N)

hbc = 197.328284
r_safe = r.clone()
r_safe[r_safe < 1e-10] = 1e-10
kappa_exp = kappa.unsqueeze(1)

# ---- 逐项计算 h_psi_g 的每一项贡献 ----
print(f"\n[2a] h_psi_g 逐项分解（单位: fm^{-1}，乘hbc后转MeV）:")
term_dg = (-df_test)                                    # 动能: -dF/dr
term_kappa = ((kappa_exp / r_safe) * f_test)             # 自旋轨道: (κ/r)F
term_vtt = (vtt * f_test)                                # 张量势
_term_YF = (YF * f_test)                                 # YF交换
_term_vps = (vps * g_test)                               # 标量势
_term_YG = (YG * g_test)                                 # YG交换
_Mg = (939.0/hbc * g_test)                               # ★ 新增质量项

terms_g = {
    '-dF/dr':      term_dg,
    '(κ/r)F':      term_kappa,
    'vtt·F':       term_vtt,
    'YF·F':        _term_YF,
    'vps·G':       _term_vps,
    'YG·G':        _term_YG,
    'M_hc·G (★)':  _Mg,
}

for name, term in terms_g.items():
    integral = (g_test * term * dr).sum().item() * hbc  # 转MeV
    print(f"  ∫G·({name})dr = {integral:+8.2f} MeV")

# 总计
h_psi_g_total = sum(terms_g.values()) * hbc
total_g_contrib = (g_test * h_psi_g_total * dr).sum().item()
print(f"  ─────────────────────────────")
print(f"  ∫G·h_psi_g dr = {total_g_contrib:+8.2f} MeV")

# ---- 逐项计算 h_psi_f 的每一项贡献 ----
print(f"\n[2b] h_psi_f 逐项分解（单位: fm^{-1}，乘hbc后转MeV）:")
term_dg2 = (dg_test)                                     # 动能: +dG/dr
term_kappa2 = ((kappa_exp / r_safe) * g_test)            # 自旋轨道: (κ/r)G
term_vtt2 = (vtt * g_test)                               # 张量势
_XG = (XG * g_test)                                      # XG交换
_vms = (vms * f_test)                                    # 标量势
_XF = (XF * f_test)                                      # XF交换
_Mf = (-939.0/hbc * f_test)                              # ★ 新增质量项

terms_f = {
    '+dG/dr':      term_dg2,
    '(κ/r)G':      term_kappa2,
    'vtt·G':       term_vtt2,
    'XG·G':        _XG,
    'vms·F':       _vms,
    'XF·F':        _XF,
    '-M_hc·F (★)': _Mf,
}

for name, term in terms_f.items():
    integral = (f_test * term * dr).sum().item() * hbc
    print(f"  ∫F·({name})dr = {integral:+8.2f} MeV")

h_psi_f_total = sum(terms_f.values()) * hbc
total_f_contrib = (f_test * h_psi_f_total * dr).sum().item()
print(f"  ─────────────────────────────")
print(f"  ∫F·h_psi_f dr = {total_f_contrib:+8.2f} MeV")

# ---- 总Rayleigh商 ----
rayleigh_numer = (total_g_contrib + total_f_contrib)
rayleigh_denom = (torch.sum((g_test**2 + f_test**2) * dr)).item()
print(f"\n[2c] Rayleigh商汇总:")
print(f"  分子(Numerator)   = {rayleigh_numer:+.2f} MeV")
print(f"  分母(Denominator) = {rayleigh_denom:.6f}")
print(f"  E_Rayleigh        = {rayleigh_numer/rayleigh_denom:+.2f} MeV")
print(f"  目标值(结合能)     ≈ -35 MeV (1s1/2 in 16O)")
print(f"  偏差              = {abs(rayleigh_numer/rayleigh_denom + 35):.1f} MeV")

# ---- 关键问题: M_hc贡献的相对大小 ----
M_contribution = (g_test * (_Mg*hbc) * dr).sum().item() + (f_test * (_Mf*hbc) * dr).sum().item()
no_M_rayleigh = rayleigh_numer - M_contribution
print(f"\n[2d] ★ 质量项影响分析:")
print(f"  M_hc总贡献       = {M_contribution:+.2f} MeV")
print(f"  无M的Rayleigh    = {no_M_rayleigh/rayleigh_denom:+.2f} MeV")
print(f"  有M的Rayleigh    = {rayleigh_numer/rayleigh_denom:+.2f} MeV")
print(f"  M引起的偏移      = {M_contribution/rayleigh_denom:+.2f} MeV")
if abs(M_contribution) > 200:
    print(f"  ⚠️ 质量项贡献过大! 可能导致E_rayleigh偏离物理区域")

# ============================================================
#  Test 3: M_hc敏感性扫描
# ============================================================
print("\n" + "="*70)
print("Test 3: M_hc系数敏感性扫描")
print("="*70)

print("\nM_coeff  |  E_Rayleigh (MeV)  |  Δ从默认值")
print("-"*50)
M_default = 939.0
for M_coef in [0, 469.5, 939.0, 1408.5, 1878.0]:
    _Mg_scan = (M_coef/hbc * g_test)
    _Mf_scan = (-M_coef/hbc * f_test)
    
    hpg = hbc * (-df_test + (kappa_exp/r_safe+vtt+YF)*f_test + (vps+YG)*g_test + _Mg_scan)
    hpf = hbc * (dg_test + (kappa_exp/r_safe+vtt+XG)*g_test + (vms+XF)*f_test + _Mf_scan)
    
    numer = (g_test*hpg + f_test*hpf * dr).sum().item()
    denom = (torch.sum((g_test**2+f_test**2)*dr)).item()
    E_val = numer/denom
    
    marker = " ← 默认" if abs(M_coef - 939.0) < 1 else ""
    print(f"  {M_coef:7.1f}  |  {E_val:+14.2f}     |  {E_val - rayleigh_numer/rayleigh_denom:+8.2f}{marker}")

# ============================================================
#  Test 4: 波函数质量对E_rayleigh的影响
# ============================================================
print("\n" + "="*70)
print("Test 4: 波函数形状对E_rayleigh的敏感性")
print("="*70)

print("\n衰减α |  peak(G) |  peak(F) |  E_Rayleigh |  M_hc贡献")
print("-"*65)
for alpha in [0.3, 0.5, 0.8, 1.2, 1.6, 2.0]:
    _g = r * torch.exp(-alpha * r)
    _f = 0.03 * r**2 * torch.exp(-alpha * r)
    _norm = torch.sqrt(torch.sum(_g**2 + _f**2) * dr)
    _g = _g / _norm; _f = _f / _norm
    
    _dg = _apply_fd_matrix(_g.squeeze(), D_fwd).unsqueeze(0)
    _df = _apply_fd_matrix(_f.squeeze(), D_bwd).unsqueeze(0)
    
    _Mg_a = (939.0/hbc * _g); _Mf_a = (-939.0/hbc * _f)
    _hpg = hbc * (-_df + (kappa_exp/r_safe+vtt+YF)*_f + (vps+YG)*_g + _Mg_a)
    _hpf = hbc * (_dg + (kappa_exp/r_safe+vtt+XG)*_g + (vms+XF)*_f + _Mf_a)
    
    _numer = (_g*_hpg + _f*_hpf * dr).sum().item()
    _denom = torch.sum((_g**2+_f**2)*dr).item()
    _E = _numer/_denom
    _M_contrib = (_g*(_Mg_a*hbc) + _f*(_Mf_a*hbc) * dr).sum().item()
    
    print(f"  {alpha:.1f}   |  {_g.abs().max():.3f}   |  {_f.abs().max():.4f}  |  {_E:+9.1f}    |  {_M_contrib:+8.1f}")

# ============================================================
#  Test 5: 导数项数值稳定性
# ============================================================
print("\n" + "="*70)
print("Test 5: 导数项数值稳定性检查")
print("="*70)

# 用已知解析函数验证差分精度
# f(r) = sin(kr), f'(r) = k*cos(kr)
k_test = 1.0
r_np = np.arange(npt) * dr
r_np[0] = 0.001
f_exact = np.sin(k_test * r_np)
df_exact = k_test * np.cos(k_test * r_np)
f_torch = torch.tensor(f_exact, device=device, dtype=torch.float32)

# 前向差分
df_fwd = _apply_fd_matrix(f_torch, D_fwd).cpu().numpy()
# 后向差分
df_bwd = _apply_fd_matrix(f_torch, D_bwd).cpu().numpy()

err_fwd = df_fwd - df_exact
err_bwd = df_bwd - df_exact

print(f"[5a] f(r)=sin(r), f'(r)=cos(r) 差分误差:")
print(f"  前向差分最大误差: |Δ|max = {np.abs(err_fwd).max():.2e} (i={np.abs(err_fwd).argmax()})")
print(f"  后向差分最大误差: |Δ|max = {np.abs(err_bwd).max():.2e} (i={np.abs(err_bwd).argmax()})")
print(f"  前向误差RMS:      {np.sqrt(np.mean(err_fwd**2)):.2e}")
print(f"  后向误差RMS:      {np.sqrt(np.mean(err_bwd**2)):.2e}")

# 检查近核区(r<1fm)的误差
mask_near = r_np < 1.0
print(f"\n[5b] 近核区(r<1fm)误差:")
print(f"  前向: |Δ|max = {np.abs(err_fwd[mask_near]).max():.2e}")
print(f"  后向: |Δ|max = {np.abs(err_bwd[mask_far := r_np > 10]).max():.2e}")
print(f"  远场区(r>10fm):")
print(f"  前向: |Δ|max = {np.abs(err_fwd[mask_far]).max():.2e}")
print(f"  后向: |Δ|max = {np.abs(err_bwd[mask_far]).max():.2e}")

# ============================================================
#  Test 6: 势场数值范围
# ============================================================
print("\n" + "="*70)
print("Test 6: 势场数据数值范围")
print("="*70)

vps_np = vps.cpu().numpy()[0]
vms_np = vms.cpu().numpy()[0]
vtt_np = vtt.cpu().numpy()[0]
r_np = r.cpu().numpy()[0]

for name, v_data in [('vps(Σ+)', vps_np), ('vms(Σ-)', vms_np), ('vtt', vtt_np)]:
    core = v_data[:int(6.0/dr)]
    tail = v_data[int(6.0/dr):]
    print(f"  {name:12s}: 全域=[{v_data.min():+.1f}, {v_data.max():+.1f}] "
          f"核心(<6fm)=[{core.mean():+.1f}±{core.std():.1f}] "
          f"远场=[{tail.min():+.1f}, {tail.max():+.1f}]")

# ============================================================
#  总结与建议
# ============================================================
print("\n" + "="*70)
print("诊断总结")
print("="*70)
print("""
根据以上测试结果，E_rayleigh震荡的可能原因按优先级排序:

  【高优先级】
  □ M_hc项贡献过大 → 导致E_rayleigh对波函数微小变化极度敏感
  □ 硬归一化(norm=const) → 波函数幅度冻结，只有形状变化驱动能量
  □ E_network梯度流被其他损失压制 → 能量头几乎不更新

  【中优先级】  
  □ F分量0.05缩放可能影响导数项平衡
  □ calc_simplified用中心差分但calc_physics用5PADF → 不一致

  【低优先级】
  □ 近核区r→0奇异性处理
  □ 远场截断效应
""")
