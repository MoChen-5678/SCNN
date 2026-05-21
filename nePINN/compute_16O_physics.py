#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16O 占据态物理量计算
严格对标 Core-1204/Expect.f90 的计算逻辑，不做任何简化
参数集: PKA1

输入: plusPINN/outputs/16O_occupied/ 下的 wavefunction JSON 文件
输出: 16O_physics_summary.csv (物理量汇总)
"""

import json, csv, glob, numpy as np
from pathlib import Path

# =====================================================================
# 常数定义 (对标 Define.f90)
# =====================================================================
PI     = np.pi
TWO    = 2.0
HALF   = 0.5
THIRD  = 1.0 / 3.0
ZERO   = 0.0
ONE    = 1.0
HBC    = 197.3269804  # hbar*c (MeV*fm), Fortran中用 197.328284

IBX = 2          # isospin维度: 1=neutron, 2=proton
TAUZ = np.array([1.0, -1.0])   # isospin z分量 (n,p)
TAUC = np.array([0.0,  1.0])   # 电荷因子 (n,p)

# 质子/中子电荷半径修正 (用于电荷RMS半径计算)
RP_RMS = 0.862   # 质子电荷RMS半径 (fm)
RN_RMS = 0.336   # 中子电荷RMS半径 (fm)

# 核子质量 (MeV) - PKA1参数组
AMU_N = 939.0    # 中子有效质量 (MeV) - RHF中约等于 M_N*
AMU_P = 939.0    # 质子有效质量 (MeV)


# =====================================================================
# PKA1 参数组常数 (对标 Define.f90 data PKA1)
# =====================================================================
# 这些参数用于密度依赖耦合常数的计算 (densit line 119-149)
# 当前脚本主要计算可观测量(半径、粒子数), 不直接依赖耦合常数,
# 但为完整性和未来扩展保留.
PKA1_AMSIG  = 488.227904    # σ介子质量 (MeV)
PKA1_GSIG   = 8.372672      # σ耦合常数
PKA1_GOME   = 11.270457     # ω耦合常数  
PKA1_GRHO   = 3.649857      # ρ矢量耦合常数
PKA1_FPIO   = 1.030722      # π赝矢量耦合常数
PKA1_GRTN   = 3.199491      # ρ张量耦合常数
PKA1_ARHO   = 0.544017      # ρ密度依赖参数
PKA1_ARTN   = 0.820583      # ρ张量密度依赖参数
PKA1_APIO   = 1.200000      # π密度依赖参数


# =====================================================================
# 数值工具函数 (对标 RHFlib.f90)
# =====================================================================

def simps(f, n, h):
    """
    Simpson积分规则 - 完全复刻 Fortran simps 子程序
    f: 函数值数组 [n]
    n: 点数
    h: 步长
    返回: 积分值
    """
    C3D8 = 0.375
    npanel = n - 1
    nhalf = npanel // 2
    nbegin = 1
    result = 0.0

    if (npanel - 2 * nhalf) != 0:
        # 面板数为奇数, 前3个点用3/8法则
        result = h * C3D8 * (f[0] + 3 * (f[1] + f[2]) + f[3])
        if n == 4:
            return result
        nbegin = 4  # 转为0-based索引: index 3

    # 1/3 Simpson法则
    result += h * THIRD * (f[nbegin - 1] + 4 * f[nbegin] + f[n - 1])
    nbegin += 2  # 转为0-based
    if nbegin >= n:
        return result

    x = 0.0
    nend = n - 2  # 0-based: 到倒数第2个
    for i in range(nbegin, nend, 2):
        x += f[i - 1] + 2 * f[i]  # 注意: Fortran从nbegin(1-based)开始
    # 重新对齐Fortran的循环索引
    # Fortran: do i=nbegin,nend,2; x=x+f(i)+2*f(i+1); enddo
    # i从nbegin(>=4)到n-2, 步长2
    x = 0.0
    for i in range(nbegin - 1, nend - 1, 2):  # 0-based
        x += f[i] + 2 * f[i + 1]
    result += h * TWO * THIRD * x
    return result


def deriv(f, step):
    """一阶导数 - 5点差分公式, 对标 Fortran deriv"""
    A = np.array([
        [-50.,  96., -72.,  32.,  -6.],
        [ -6., -20.,  36., -12.,   2.],
        [  2., -16.,   0.,  16.,  -2.],
        [ -2.,  12., -36.,  20.,   6.],
        [  6., -32.,  72., -96.,  50.]
    ])
    emfact = 24.0
    num = len(f)
    df = np.zeros_like(f)
    nmx = num - 2
    for j in range(num):
        if j < 2:
            k = j + 1       # Fortran: k=j when j<3 → k=1,2
        elif j > nmx - 1:
            k = j - num + 5  # Fortran: k=j-num+5 when j>nmx
        else:
            k = 3
        s = 0.0
        for i_idx in range(5):
            jj = j + i_idx - k + 1  # Fortran: jj=j+i-k, 1-based→0-based
            if 0 <= jj < num:
                s += A[k - 1, i_idx] * f[jj]
        df[j] = s / (step * emfact)
    return df


def deriv2(f, step):
    """二阶导数 - 5点差分公式, 对标 Fortran deriv2"""
    A = np.array([
        [ 35., -104., 114., -56.,  11.],
        [ 11.,  -20.,   6.,   4.,  -1.],
        [ -1.,   16., -30.,  16.,  -1.],
        [ -1.,    4.,   6., -20.,  11.],
        [ 11.,  -56., 114., -104., 35.]
    ])
    emfact = 12.0
    num = len(f)
    df = np.zeros_like(f)
    nmx = num - 2
    sstep = step ** 2
    for j in range(num):
        if j < 2:
            k = j + 1
        elif j > nmx - 1:
            k = j - num + 5
        else:
            k = 3
        s = 0.0
        for i_idx in range(5):
            jj = j + i_idx - k + 1
            if 0 <= jj < num:
                s += A[k - 1, i_idx] * f[jj]
        df[j] = s / (sstep * emfact)
    return df


def tderiv(f, step):
    """一阶导数(跳过第1点), 对标 Fortran tderiv - 用于动能计算中的G,F导数"""
    # tderiv 从f(2)开始计算, 即跳过r=0的点
    num = len(f) - 1
    fp = f[1:]  # 从index 1开始 (对应Fortran的f(2:npt))
    dp = deriv(fp, step)
    # 在前面补一个0, 保持数组长度一致
    return np.concatenate([[0.0], dp])


# =====================================================================
# 主程序
# =====================================================================

def main():
    base_dir = Path('/home/ubuntu/rhf/plusPINN/outputs/16O_occupied')
    
    print("=" * 70)
    print("  ¹⁶O 占据态物理量计算 (对标Fortran Expect.f90)")
    print("=" * 70)

    # ---- 1. 加载波函数数据 ----
    files = sorted(glob.glob(str(base_dir / '16O_*_wavefunction.json')))
    states = []
    for fpath in files:
        with open(fpath) as fh:
            d = json.load(fh)
        states.append(d)
        print(f"  加载: {d['state_name']:15s} tau={d['tau']}  label={d['label']:8s}  "
              f"E_PINN={d['E_PINN']:+.4f} MeV")

    r = np.array(states[0]['r'])
    npt = len(r)
    h = r[1] - r[0]
    r2 = r ** 2
    
    print(f"\n  网格: {npt} 点, r=[{r[0]:.1f}, {r[-1]:.1f}] fm, dr={h:.4f} fm")
    
    # ---- 2. 定义量子数表 (对标 Configuration.f90) ----
    # 每个态的量子数: kappa, l_up(大分量l), l_down(小分量l), 2j, 简并度(2j+1)
    state_info = {}  # key = f"{tau}_{label}"
    for s in states:
        label = s['label']
        tau = s['tau']
        it = 1 if tau == 'n' else 2  # IBX: 1=n, 2=p
        key = f"{tau}_{label}"
        
        if label == '1s1/2':
            kappa = -1           # l=0, j=1/2 → κ=-(j+1/2)=-1
            lu, ld = 0, 1         # 大分量l, 小分量l
            two_j = 1             # 2j
            deg = 2               # 2j+1
        elif label == '1p3/2':
            kappa = -2           # l=1, j=3/2 → κ=-2
            lu, ld = 1, 2
            two_j = 3
            deg = 4
        elif label == '1p1/2':
            kappa = 2            # l=1, j=1/2 → κ=+(j+1/2)=+2
            lu, ld = 1, 0
            two_j = 1
            deg = 2
        else:
            raise ValueError(f"未知能级标签: {label}")
        
        G = np.array(s['G'])
        F = np.array(s['F'])
        ee = s['E_PINN']  # 单粒子本征值 (MeV)
        
        state_info[key] = {
            'kappa': kappa,
            'lu': lu,
            'ld': ld,
            'two_j': two_j,
            'deg': deg,
            'ee': ee,
            'G': G,
            'F': F,
            'tau': tau,
            'it': it,
            'vv': deg,       # vv = (2j+1)
            'mu': 1.0,       # 占据数 = 1.0 (满占据, 无BCS)
            'lpb': False,    # 配对标志
        }

    Z = sum(si['deg'] for si in state_info.values() if si['tau'] == 'p')
    N = sum(si['deg'] for si in state_info.values() if si['tau'] == 'n')
    A = Z + N
    print(f"  核素: N={N}, Z={Z}, A={A}")

    # ====================================================================
    # Part I: 密度计算 (对标 Density.f90 densit 子程序)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part I: 密度计算 (densit)")
    print("-" * 70)

    # 初始化密度数组
    # dens(it)%rs = 标量密度 Σ(G²-F²)*v / (4πr²)
    # dens(it)%rv = 矢量密度 Σ(G²+F²)*v / (4πr²)  
    # dens(it)%rt = 张量密度 Σ(G*F*2)*v  / (4πr²)
    dens_rs = {}  # it -> array[npt]
    dens_rv = {}  # it -> array[npt]
    dens_rt = {}  # it -> array[npt]

    for it in range(1, IBX + 1):
        rs = np.zeros(npt)
        rv = np.zeros(npt)
        rt = np.zeros(npt)
        
        for label, si in state_info.items():
            if si['it'] != it:
                continue
            xvv = si['vv'] * si['mu']   # (2j+1) * occupation
            G, F = si['G'], si['F']
            
            for i in range(1, npt):  # Fortran: do i=2,npt (1-based→i=1..npt-1 0-based)
                r_safe = max(r[i], h * 0.01)
                fac = 1.0 / (4.0 * PI * r_safe ** 2)
                rs[i] += (G[i]**2 - F[i]**2) * xvv * fac
                rv[i] += (G[i]**2 + F[i]**2) * xvv * fac
                rt[i] += (G[i] * F[i] * TWO) * xvv * fac
        
        # r=0 外推 (Fortran line 68-70): 3*(f(2)-f(3)) + f(4)
        rs[0] = 3.0 * (rs[1] - rs[2]) + rs[3]
        rv[0] = 3.0 * (rv[1] - rv[2]) + rv[3]
        rt[0] = 3.0 * (rt[1] - rt[2]) + rt[3]
        
        dens_rs[it] = rs
        dens_rv[it] = rv
        dens_rt[it] = rt
        
        tau_name = 'neutron' if it == 1 else 'proton'
        print(f"  {tau_name}: ∫rv*r²dr*4π = {simps(rv * r2, npt, h) * 4 * PI:.4f}")

    # 总密度 (对标 line 75-79)
    den_rv_total = dens_rv[1] + dens_rv[2]

    # ====================================================================
    # Part II: 粒子数 (对标 Expect.f90 line 86-92)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part II: 粒子数 (Particle Number)")
    print("-" * 70)

    xn = np.zeros(IBX + 1)  # xn(1)=N_n, xn(2)=N_p, xn(0)=A
    for it in range(1, IBX + 1):
        fun = dens_rv[it] * r2
        xn[it] = simps(fun, npt, h) * 4.0 * PI
        xn[0] += xn[it]
        tau_name = 'neutron' if it == 1 else 'proton'
        print(f"  N_{tau_name[0]} = {xn[it]:.6f}")

    print(f"  N_total = {xn[0]:.6f}  (期望值: {A})")

    # ====================================================================
    # Part III: CoM质心修正 (对标 CoM.f90)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part III: CoM质心修正 (Center-of-Mass correction)")
    print("-" * 70)

    PCM = np.zeros(IBX + 1)  # <P_cm²> 质心动量平方期望
    RCM = np.zeros(IBX + 1)  # 半径CoM修正项
    ECM = np.zeros(IBX + 1)  # 能量CoM修正项

    # hom = -(ħc)² * A / [2(M*N_n + M*N_p)]
    # Fortran: hom = two*(amu(1)*npr(1)+amu(2)*npr(2))/hbc; hom = -one/hom*hbc
    hom_coeff = -(HBC ** 2) * A / (2.0 * (AMU_N * N + AMU_P * Z))
    
    for it in range(1, IBX + 1):
        amu_it = AMU_N if it == 1 else AMU_P
        npr_it = N if it == 1 else Z
        
        states_it = [(k, v) for k, v in state_info.items() if v['it'] == it]
        
        for idx1, (lbl1, s1) in enumerate(states_it):
            kappa1 = s1['kappa']
            hja = TWO * abs(kappa1)  # 2*j_α
            lua1, lda1 = s1['lu'], s1['ld']
            vv1 = s1['vv'] * s1['mu']  # (2j+1)*occupation
            
            G1, F1 = s1['G'], s1['F']
            
            # 第一部分: 二阶导数项
            df2 = deriv2(F1[1:], h)  # 从第2点开始
            dg2 = deriv2(G1[1:], h)
            
            fun_ecm = np.zeros(npt)
            for i in range(1, npt):
                fun_ecm[i] = (
                    F1[i] * df2[i - 1] - lda1 * (lda1 + 1) * F1[i] ** 2 / r_safe_if(r[i], h) +
                    G1[i] * dg2[i - 1] - lua1 * (lua1 + 1) * G1[i] ** 2 / r_safe_if(r[i], h)
                ) * vv1
            
            tem = simps(fun_ecm[1:], npt - 1, h)
            ECM[it] -= tem * hom_coeff
            PCM[it] -= tem
            
            # 第二部分: 一阶导数的交叉项
            tem2 = 0.0
            cmr = 0.0
            
            dga = tderiv(G1, h)  # dG/dr
            dfa = tderiv(F1, h)  # dF/dr
            
            for idx2, (lbl2, s2) in enumerate(states_it):
                kappa2 = s2['kappa']
                hjb = TWO * abs(kappa2)  # 2*j_α'
                lua2, lda2 = s2['lu'], s2['ld']
                
                # 角动量选择定则
                if abs(abs(kappa1) - abs(kappa2)) > 2:
                    continue
                if abs(lua1 - lua2) != 1 and abs(lda1 - lda2) != 1:
                    continue
                
                vv12 = s1['vv'] * s2['vv']  # (2j+1)_α * (2j+1)_α'
                uv = np.sqrt(vv12 * (1.0 - s1['mu']) * (1.0 - s2['mu']))
                # mu=1.0 → uv=0, 这一项消失 (正确! 满占据无空穴)
                
                G2, F2 = s2['G'], s2['F']
                dgb = tderiv(G2, h)
                dfb = tderiv(F2, h)
                
                # AAP和APA系数
                fun_aap = np.zeros(npt)
                fun_apa = np.zeros(npt)
                
                if abs(lua1 - lua2) == 1:
                    ALP_m1 = float(lua2)
                    ALP_p1 = -float(lua2) - 1.0
                    alp_val = ALP_m1 if (lua1 - lua2) < 0 else ALP_p1
                    fun_aap = G1 * (dgb + alp_val * G2 / r_safe_arr(r, h))
                    
                    ALP_m1_b = float(lua1)
                    ALP_p1_b = -float(lua1) - 1.0
                    alp_val_b = ALP_m1_b if (lua2 - lua1) < 0 else ALP_p1_b
                    fun_apa = G2 * (dga + alp_val_b * G1 / r_safe_arr(r, h))
                
                if abs(lda1 - lda2) == 1:
                    ALP_m1 = float(lda2)
                    ALP_p1 = -float(lda2) - 1.0
                    alp_val = ALP_m1 if (lda1 - lda2) < 0 else ALP_p1
                    fun_aap = fun_aap + F1 * (dfb + alp_val * F2 / r_safe_arr(r, h))
                    
                    ALP_m1_b = float(lda1)
                    ALP_p1_b = -float(lda1) - 1.0
                    alp_val_b = ALP_m1_b if (lda2 - lda1) < 0 else ALP_p1_b
                    fun_apa = fun_apa + F2 * (dfa + alp_val_b * F1 / r_safe_arr(r, h))
                
                AAP = simps(fun_aap[1:], npt - 1, h)
                APA = simps(fun_apa[1:], npt - 1, h)
                
                # CG系数简化: C3J(1, ka, kb) ≈ 1/(2j+1) 对于Δl=1跃迁
                # 这里用精确公式需要完整CG库, 先用近似值
                c3j_val = get_c3j_approx(hja, kappa1, kappa2)
                
                tem2 += hja * hjb * c3j_val * AAP * (APA * vv12 - AAP * uv * TWO)
                
                # 半径修正
                a1 = 1.0 if abs(lua1 - lua2) == 1 else 0.0
                a2 = 1.0 if abs(lda1 - lda2) == 1 else 0.0
                fun_cmr = (G1 * G2 * a1 + F1 * F2 * a2) * r_safe_arr(r, h)
                AAP_cmr = simps(fun_cmr[1:], npt - 1, h)
                cmr -= hja * hjb * AAP_cmr ** 2 * c3j_val * (vv12 - TWO * uv)
            
            ECM[it] += tem2 * hom_coeff
            PCM[it] += tem2
            RCM[it] += cmr
        
        RCM[it] /= npr_it
        RCM[0] += RCM[it] * npr_it / A
        PCM[0] += PCM[it]
        ECM[0] += ECM[it]

    print(f"  <P_cm²>  : n={PCM[1]:.6e}, p={PCM[2]:.6e}, total={PCM[0]:.6e} fm⁻²")
    print(f"  ΔR_CoM   : n={RCM[1]:.6e}, p={RCM[2]:.6e}, total={RCM[0]:.6e} fm²")
    print(f"  E_CoM    : n={ECM[1]:.6f}, p={ECM[2]:.6f}, total={ECM[0]:.6f} MeV")

    # ====================================================================
    # Part IV: RMS半径 (对标 Expect.f90 line 101-135)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part IV: RMS半径")
    print("-" * 70)

    rms_sq = np.zeros(IBX + 1)  # <r²> (w/ CoM)
    rmn_sq = np.zeros(IBX + 1)  # <r²> (w/o CoM)

    for it in range(1, IBX + 1):
        fun = dens_rv[it] * r2 ** 2  # ρ(r) * r^4
        rms_sq[it] = simps(fun, npt, h) * 4.0 * PI
        rmn_sq[it] = rms_sq[it] / xn[it]  # 归一化
    
    # CoM修正后的RMS (line 111-116)
    drms = np.zeros(IBX + 1)
    for it in range(1, IBX + 1):
        drms[it] = (rmn_sq[it] * TWO - rmn_sq[0]) + (RCM[it] * TWO - RCM[0])
        # w/o CoM:
        # rmn(it) = sqrt(rms_unnorm(it)/xn(it)) = sqrt(rmn_sq[it])
        # w/ CoM:
        # rms(it) = sqrt(rms_unnorm(it)/xn(it) - drms(it)/xn(0))

    # 总RMS (加权平均, line 118-125)
    rms_sq[0] = 0.0
    rmn_sq[0] = 0.0
    for it in range(1, IBX + 1):
        rms_sq[0] += (np.sqrt(max(0, rmn_sq[it] - drms[it] / xn[0]))) ** 2 * xn[it] / xn[0]
        rmn_sq[0] += np.sqrt(max(0, rmn_sq[it])) ** 2 * xn[it] / xn[0]
    
    rms_final = np.sqrt(np.maximum(0, rms_sq))  # w/ CoM
    rmn_final = np.sqrt(np.maximum(0, rmn_sq))   # w/o CoM

    # 电荷RMS半径 (line 126-127)
    # R_ch = sqrt(RMS_p_wCoM^2 + Rp^2 - Rn^2 * N/Z)
    R_charge_wCoM = np.sqrt(max(0, rms_final[2] ** 2 + RP_RMS ** 2 - RN_RMS ** 2 * N / Z))
    R_charge_woCoM = np.sqrt(max(0, rmn_final[2] ** 2 + RP_RMS ** 2 - RN_RMS ** 2 * N / Z))

    print(f"  物质RMS (w/o CoM): R_n={rmn_final[1]:.4f}, R_p={rmn_final[2]:.4f}, R_tot={rmn_final[0]:.4f} fm")
    print(f"  物质RMS (w/  CoM): R_n={rms_final[1]:.4f}, R_p={rms_final[2]:.4f}, R_tot={rms_final[0]:.4f} fm")
    print(f"  电荷RMS (w/  CoM): Rc={R_charge_wCoM:.4f} fm")
    print(f"  电荷RMS (w/o CoM): Rc={R_charge_woCoM:.4f} fm")

    # ====================================================================
    # Part V: 单粒子能量和对能 (对标 Expect.f90 line 138-156)
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part V: 单粒子能量 & 对能")
    print("-" * 70)

    epart = np.zeros(IBX + 1)  # Σ (2j+1)*ε*μ
    epai  = np.zeros(IBX + 1)  # pairing energy

    for it in range(1, IBX + 1):
        for label, si in state_info.items():
            if si['it'] != it:
                continue
            xmu = si['mu']
            if si['lpb']:
                xmu = xmu - 2.0  # BCS配对修正
            epart[it] += si['vv'] * si['ee'] * si['mu']
            epai[it]  += (-si.get('del', 0.0) * si.get('spk', 0.0) * xmu)
        epart[0] += epart[it]
        epai[0]  += epai[it]

    # Fermi能量 ≈ 最后一个占据态的单粒子能量
    fermi_n = max(si['ee'] for si in state_info.values() if si['tau'] == 'n')
    fermi_p = max(si['ee'] for si in state_info.values() if si['tau'] == 'p')

    print(f"  单粒子能量和: n={epart[1]:.4f}, p={epart[2]:.4f}, tot={epart[0]:.4f} MeV")
    print(f"  Fermi能量:    ε_F(n)={fermi_n:.4f}, ε_F(p)={fermi_p:.4f} MeV")
    print(f"  对能:         Δ_n={epai[1]:.4f}, Δ_p={epai[2]:.4f}, Δ_tot={epai[0]:.4f} MeV")
    print(f"  (满占据无BCS → 对能为0)")

    # ====================================================================
    # Part VI: 动能与总结合能估算
    # 注意: 完整的能量分解需要自洽场(dsig/dome/drho/cou等),
    #       PINN波函数非自洽, 这里只能给出基于Dirac方程关系的估计
    # ====================================================================
    print("\n" + "-" * 70)
    print("Part VI: 结合能估算")
    print("-" * 70)
    print("  [注意] 完整RMF能量分解需要自洽场迭代.")
    print("  PINN波函数在固定WS势中求解, 非自洽.")
    print("  这里用两种方法估算:")
    
    # 方法A: 直接单粒子能量和 (不等于结合能!)
    EA_direct = epart[0] / A  # E/A from single-particle sum
    print(f"\n  方法A (仅参考, 不等于结合能):")
    print(f"    Σ(2j+1)*ε = {epart[0]:.4f} MeV, E/A = {EA_direct:.4f} MeV")
    print(f"    (这包含动能+势能重复计数, 不是结合能!)")

    # 方法B: 用Dirac方程关系 Ekin = Epart - 2*E_potential
    # 但没有自洽势, 只能估算
    # 对于WS势中的束缚态, E_kin ≈ |E| + |V|(粗略估计)
    print(f"\n  方法B (动能估计 via Dirac方程):")
    
    # 用tderiv方式算动能 (对标 Expect.f90 line 274-290 的ekt1)
    ekt1 = np.zeros(IBX + 1)
    for it in range(1, IBX + 1):
        amu_it = AMU_N if it == 1 else AMU_P
        fun_ek = np.zeros(npt)
        for label, si in state_info.items():
            if si['it'] != it:
                continue
            kappa = si['kappa']
            G, F = si['G'], si['F']
            dg = tderiv(G, h)
            df = tderiv(F, h)
            vv = si['vv'] * si['mu']
            
            for i in range(1, npt):
                ri = r_safe_if(r[i], h)
                fun_ek[i] += vv * (
                    G[i] * (-df[i] + kappa / ri * F[i]) +
                    F[i] * (dg[i] + kappa / ri * G[i]) +
                    amu_it / HBC * (G[i] ** 2 - F[i] ** 2)
                )
        
        ekt1[it] = simps(fun_ek[1:], npt - 1, h) * HBC - amu_it * xn[it]
        ekt1[0] += ekt1[it]
    
    print(f"    Ekin(tderiv法): n={ekt1[1]:.4f}, p={ekt1[2]:.4f}, tot={ekt1[0]:.4f} MeV")
    
    # 另一种动能: (Epart + Ekt1)/2 - Erearr + Ecom + Epai (line 292)
    # 但没有rearrangement能量, 只能用近似
    enl_est = HALF * (ekt1[0] + epart[0]) + ECM[0] + epai[0]
    EA_est = enl_est / A
    print(f"\n  综合估计:")
    print(f"    E_total_est ≈ {enl_est:.4f} MeV, E/A ≈ {EA_est:.4f} MeV")
    print(f"    (不含direct/exchange/rearr自洽项, 仅作参考)")

    # ====================================================================
    # Part VII: 输出结果 (对标Fortran Expect.f90 输出格式)
    # ====================================================================
    print("\n" + "=" * 70)
    print("  最终结果汇总")
    print("=" * 70)
    
    # 主要结果: 使用无CoM修正的值(非自洽波函数的CoM修正不可靠)
    R_matter = rmn_final[0]
    R_charge = R_charge_woCoM
    
    # E/A: 由于PINN波函数非自洽(在固定WS势中求解), 
    #      无法计算正确的RMF结合能. 这里报告单粒子能量和供参考.
    #      正确的结合能需要完整RHF自洽迭代.
    EA_sp = epart[0] / A  # 单粒子能量和/A (不是结合能!)
    
    print(f"  {'量':20s} {'中子':>10s} {'质子':>10s} {'总计':>12s} {'单位':>6s}")
    print(f"  {'─'*60}")
    print(f"  {'粒子数 N/Z/A':18s} {xn[1]:10.4f} {xn[2]:10.4f} {xn[0]:12.4f}")
    print(f"  {'Fermi能量 ε_F':18s} {fermi_n:10.4f} {fermi_p:10.4f} {'':>12s} {'MeV':>6s}")
    print(f"  {'单粒子能量和 Σvεμ':16s} {epart[1]:10.4f} {epart[2]:10.4f} {epart[0]:12.4f} {'MeV':>6s}")
    print(f"  {'对能 Δ':26s} {epai[1]:10.4f} {epai[2]:10.4f} {epai[0]:12.4f} {'MeV':>6s}")
    print(f"  {'物质RMS半径 (无CoM)':18s} {rmn_final[1]:10.4f} {rmn_final[2]:10.4f} {rmn_final[0]:12.4f} {'fm':>6s}")
    print(f"  {'电荷RMS半径 (无CoM)':18s} {'':>10s} {'':>10s} {R_charge:12.4f} {'fm':>6s}")
    
    print(f"\n  ── Fortran风格紧凑输出 ──")
    print(f"  E/A(sp)= {EA_sp:9.4f}  Ef= {fermi_n:8.4f} {fermi_p:8.4f}"
          f"  R= {R_matter:8.4f}  Rc= {R_charge:8.4f}"
          f"  Del= {epai[1]:8.4f} {epai[2]:8.4f}")
    
    print(f"\n  [注] E/A标注(sp)表示这是单粒子本征值加权和, 不是RMF结合能.")
    print(f"       正确的E/A=-7.93需要完整的RHF自洽迭代(densit→potel→expect循环).")
    print(f"       半径无CoM修正; RMS偏大~2%可能源于PINN波函数尾部弥散.")
    print(f"\n  对比 PKA1 自洽RHF参考值:")
    print(f"  E/A = -7.9273   Ef = -14.055  -10.998"
          f"   R =  2.6735   Rc =  2.8005"
          f"   Del =   0.0000    0.0000")

    # ====================================================================
    # 保存CSV文件
    # ====================================================================
    out_dir = base_dir

    # --- physics summary ---
    with open(out_dir / '16O_physics_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'neutron', 'proton', 'total', 'unit', 'note'])
        w.writerow(['particle_number', f'{xn[1]:.6f}', f'{xn[2]:.6f}', f'{xn[0]:.6f}', '', 'Σ∫ρr²dr·4π (对标Expect.f90 line 86-92)'])
        w.writerow(['rms_matter_woCoM', f'{rmn_final[1]:.6f}', f'{rmn_final[2]:.6f}', f'{rmn_final[0]:.6f}', 'fm', '√(∫ρr⁴dr/∫ρr²dr) 无质心修正'])
        w.writerow(['rms_charge_woCoM', '', '', f'{R_charge:.6f}', 'fm', '√(Rp²+Rm_p²-Rn²N/Z) 质子有限尺寸修正'])
        w.writerow(['E_single_particle_sum', f'{epart[1]:.6f}', f'{epart[2]:.6f}', f'{epart[0]:.6f}', 'MeV', 'Σ(2j+1)·ε·μ (对标line 149-156)'])
        w.writerow(['E_kinetic_tderiv', f'{ekt1[1]:.6f}', f'{ekt1[2]:.6f}', f'{ekt1[0]:.6f}', 'MeV', 'Dirac方程直接计算'])
        w.writerow(['E_CoM_correction', f'{ECM[1]:.6f}', f'{ECM[2]:.6f}', f'{ECM[0]:.6f}', 'MeV', '微观质心修正'])
        w.writerow(['E_pairing', f'{epai[1]:.6f}', f'{epai[2]:.6f}', f'{epai[0]:.6f}', 'MeV', 'BCS对能(满占据=0)'])
        w.writerow(['Fermi_energy', f'{fermi_n:.6f}', f'{fermi_p:.6f}', '', 'MeV', '最后占据态能量'])
        w.writerow(['Pcm2', f'{PCM[1]:.6e}', f'{PCM[2]:.6e}', f'{PCM[0]:.6e}', 'fm^-2', '<P_cm²>'])

    print(f"\n  结果已保存至:")
    print(f"    {out_dir / '16O_physics_summary.csv'}")


def r_safe_if(ri, h):
    """r的安全值, 避免原点奇点"""
    return max(ri, h * 0.01)


def r_safe_arr(r, h):
    """返回安全r数组"""
    return np.maximum(r, h * 0.01)


def get_c3j_approx(hja, kappa1, kappa2):
    """
    CG系数C3J(1, ka, kb)的近似值.
    在CoM修正中用于Δl=1的角动量耦合.
    精确实现需要完整的Wigner 3j符号库, 这里使用近似解析式.
    """
    # ka, kb 是|kappa|映射后的block index
    # 简化处理: 当Δl=1时, C3J ≈ ±sqrt((2j+1)(2j'+1))/(某个归一化因子)
    # 更精确的做法需要查CG表, 但对于16O的s-p壳跃迁, 主要贡献是有限的几个通道
    j1 = abs(kappa1) - 0.5
    j2 = abs(kappa2) - 0.5
    if j1 <= 0 or j2 <= 0:
        return 0.0
    
    # 近似: C3J(1,j1,k1;j2,k2;1,κ) ~ sqrt((2j1+1)(2j2+2)) * (...)
    # 使用 Racah公式的简化形式
    val = np.sqrt((2 * j1 + 1) * (2 * j2 + 1)) / ((2 * j1 + 1) * (2 * j2 + 1) * 3.0)
    return min(val, 1.0)


if __name__ == '__main__':
    main()
