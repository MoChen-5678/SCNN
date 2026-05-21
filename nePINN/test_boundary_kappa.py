"""
用量子数表.csv 逐条验证 get_angular_momenta() 的正确性.
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/rhf/plusPINN')
from boundary_conditions import get_angular_momenta

# ── 你的量子数表 (从CSV) ──
# 格式: (态标记, 轨道角动量l, 总角动量j, 宇称π, 狄拉克κ)
states = [
    ("1s.1/2", 0, "1/2", "+", -1),
    ("2s.1/2", 0, "1/2", "+", -1),
    ("3s.1/2", 0, "1/2", "+", -1),
    ("1p.3/2", 1, "3/2", "-", -2),
    ("2p.3/2", 1, "3/2", "-", -2),
    ("1d.5/2", 2, "5/2", "+", -3),
    ("2d.5/2", 2, "5/2", "+", -3),
    ("1f.7/2", 3, "7/2", "-", -4),
    ("1p.1/2", 1, "1/2", "-", +1),
    ("2p.1/2", 1, "1/2", "-", +1),
    ("1d.3/2", 2, "3/2", "+", +2),
    ("2d.3/2", 2, "3/2", "+", +2),
    ("1f.5/2", 3, "5/2", "-", +3),
    ("1g.7/2", 4, "7/2", "+", +4),
]

print(f"{'态标记':>8s} | {'l(表)':>4s} | {'j':>4s} | {'π':>2s} | {'κ':>3s} | "
      f"{'l_u(算)':>6s} | {'l_d(算)':>6s} | G~r^{'':>3s} | F~r^{'':>3s} | {'状态'}")
print("─" * 95)

all_ok = True
for label, l_tab, j_str, parity, kappa in states:
    l_u, l_d = get_angular_momenta(kappa)
    
    # 验证: l_u 应该等于表中的轨道角动量 l
    match_lu = (l_u == l_tab)
    
    # r→0 渐近行为指数
    g_power = l_u + 1   # G ~ r^{l_u+1}
    f_power = l_d + 1   # F ~ r^{l_d+1}
    
    status = "✓" if match_lu else "✗ WRONG"
    if not match_lu:
        all_ok = False
    
    print(f"{label:>8s} | {l_tab:>4d} | {j_str:>4s} | {parity:>2s} | {kappa:+>3d} | "
          f"{l_u:>6d} | {l_d:>6d} | {g_power:>3d}  | {f_power:>3d}  | {status}")

print("\n" + "=" * 95)
if all_ok:
    print("全部通过 ✓ — get_angular_momenta() 与量子数表一致")
else:
    print("有错误 ✗ — 见上表")
print()

# ★ 额外检查: 边界条件中r→0行为是否真的用了这些值
print("=" * 95)
print("边界条件 r→0 行为检查:")
print(f"{'态标记':>8s} | {'κ':>3s} | 预期 G(r→0)~ | 预期 F(r→0)~")
print("─" * 60)

DR = 0.10
for label, l_tab, j_str, parity, kappa in [("1s.1/2",0,"1/2","+",-1), 
                                            ("2s.1/2",0,"1/2","+",-1),
                                            ("1p.3/2",1,"3/2","-",-2),
                                            ("1p.1/2",1,"1/2","-",+1),
                                            ("1d.3/2",2,"3/2","+",+2)]:
    l_u, l_d = get_angular_momenta(kappa)
    print(f"{label:>8s} | {kappa:+>3d} | r^{l_u+1}(l={l_u}{['s','p','d','f','g'][l_u]}) | r^{l_d+1}(l={l_d}{['s','p','d','f','g'][l_d]})")
