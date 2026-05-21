"""
PINN-RHF 工具函数模块

功能:
  1. 加载 Fortran PKA1 参考数据用于对比验证
  2. 绘制 PINN vs Fortran 对比图表
  3. 数值诊断工具
  4. 数据格式转换
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from config import DR, NPT, R_GRID, ISOTOPE_CONFIG


# ═══════════════════════════════════════════════════════════════
#   Fortran 数据加载
# ═══════════════════════════════════════════════════════════════

def _fix_fortran_float(s):
    """修复 Fortran 的指数格式 (如 1.0-100 → 1.0e-100)"""
    s = str(s).strip()
    # 匹配数字后面紧跟负号的科学计数法异常
    m = re.match(r'^([+\-]?(?:\d+\.?\d*|\.\d+))[dD]([+\-]?\d+)$', s)
    if m:
        return f"{m.group(1)}e{m.group(2)}"
    # 处理 1.23-456 格式
    m = re.match(r'^([+\-]?(?:\d+\.?\d*|\.\d+))([+\-]\d+)$', s)
    if m and '.' in m.group(1):
        return f"{m.group(1)}e{m.group(2)}"
    return s


def load_pkal_wavefunction(pka1_path, npt=None):
    """
    加载 PKA1 文件中的收敛态波函数。
    
    PKA1 文件是 Core-1204 Fortran 代码输出的最终收敛波函数，
    格式通常为:  列0=r  列1=G(r)  列2=F(r)

    参数:
        pka1_path: str, PKA1 文件的完整路径
        npt: int, 目标网格点数 (可选, 用于插值或截断)
    返回:
        dict: {
            'r': (N,) 径向坐标 (fm),
            'g': (N,) 大分量,
            'f': (N,) 小分量,
            'npt': int, 实际点数,
        }
    """
    if not os.path.exists(pka1_path):
        raise FileNotFoundError(f"PKA1文件不存在: {pka1_path}")

    data = np.loadtxt(pka1_path, comments='#',
                     converters={1: _fix_fortran_float, 2: _fix_fortran_float})

    r = data[:, 0]
    g = data[:, 1]
    f = data[:, 2]

    if npt is not None:
        # 插值到目标网格
        from scipy.interpolate import interp1d
        target_r = R_GRID[:npt]

        # 使用线性插值 (足够平滑)
        g_interp = interp1d(r, g, kind='linear', fill_value='extrapolate')(target_r)
        f_interp = interp1d(r, f, kind='linear', fill_value='extrapolate')(target_r)

        return {'r': target_r, 'g': g_interp, 'f': f_interp, 'npt': npt}

    return {'r': r, 'g': g, 'f': f, 'npt': len(r)}


def find_reference_data(base_dir, isotope='16O'):
    """
    自动搜索 Fortran 参考数据文件。
    
    搜索路径优先级:
      1. base_dir/results/{isotope}/PKA1
      2. ../SCNN/data/{isotope}/*/PKA1
      3. base_dir 下递归查找 PKA1 文件

    参数:
        base_dir: str, 项目根目录
        isotope: str, 核素名称
    返回:
        dict: {state_name: file_path} 或 {} (如果未找到)
    """
    results = {}
    
    # 常见搜索路径
    search_paths = [
        os.path.join(base_dir, 'results', isotope),
        os.path.join(base_dir, '..', 'SCNN', 'data', isotope),
    ]

    for spath in search_paths:
        if not os.path.isdir(spath):
            continue
        for root, dirs, files in os.walk(spath):
            for fname in files:
                if 'PKA1' in fname.upper() or 'pka1' in fname.lower():
                    # 尝试从文件名推断态标识符
                    full_path = os.path.join(root, fname)
                    # 用目录名或文件名作为 key
                    state_key = os.path.basename(root)
                    if state_key == isotope:
                        state_key = os.path.splitext(fname)[0]
                    results[state_key] = full_path

    return results


# ═══════════════════════════════════════════════════════════════
#   对比绘图
# ═══════════════════════════════════════════════════════════════

def plot_comparison(pinn_result, ref_data=None, title='PINN vs Reference',
                   save_path=None, show=True):
    """
    绘制 PINN 解与参考解的详细对比图。
    
    参数:
        pinn_result: dict from MVPSolver.evaluate()
        ref_data: dict from load_pkal_wavefunction(), 或 None
        title: str, 图表标题
        save_path: str, 保存路径 (None 则不保存)
        show: bool, 是否显示交互窗口
    """
    r_pinn = pinn_result['r']
    g_pinn = pinn_result['g']
    f_pinn = pinn_result['f']

    has_ref = ref_data is not None

    ncols = 3 if has_ref else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

    # 1. G 分量对比
    ax = axes[0]
    ax.plot(r_pinn, g_pinn, 'b-', linewidth=2, label='PINN $G$')
    if has_ref:
        ax.plot(ref_data['r'], ref_data['g'], 'r--', linewidth=1.5, alpha=0.8, label='Ref $G$')
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('$G(r)$')
    ax.set_title('Large Component')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(20, r_pinn[-1])])

    # 2. F 分量对比
    ax = axes[1]
    ax.plot(r_pinn, f_pinn, 'b-', linewidth=2, label='PINN $F$')
    if has_ref:
        ax.plot(ref_data['r'], ref_data['f'], 'r--', linewidth=1.5, alpha=0.8, label='Ref $F$')
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('$F(r)$')
    ax.set_title('Small Component')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(20, r_pinn[-1])])

    # 3. 差异/误差分析 (如果有参考解)
    if has_ref:
        ax = axes[2]
        
        # 将 PINN 插值到参考解的网格上 (如果不同)
        if len(r_pinn) != len(ref_data['r']):
            from scipy.interpolate import interp1d
            g_on_ref_grid = interp1d(r_pinn, g_pinn, kind='linear', fill_value='extrapolate')(ref_data['r'])
            f_on_ref_grid = interp1d(r_pinn, f_pinn, kind='linear', fill_value='extrapolate')(ref_data['r'])
        else:
            g_on_ref_grid = g_pinn
            f_on_ref_grid = f_pinn

        # 相对误差 (在非零区域)
        eps = 1e-10
        g_rel_err = (g_on_ref_grid - ref_data['g']) / (np.abs(ref_data['g']) + eps)
        f_rel_err = (f_on_ref_grid - ref_data['f']) / (np.abs(ref_data['f']) + eps)

        ax.semilogy(ref_data['r'], np.abs(g_rel_err) + 1e-10, 'b-', label='$|\\Delta G/G|$')
        ax.semilogy(ref_data['r'], np.abs(f_rel_err) + 1e-10, 'r-', label='$|\\Delta F/F|$')
        ax.set_xlabel('r (fm)')
        ax.set_ylabel('Relative Error')
        ax.set_title('Relative Error')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 打印数值指标
        g_l2 = np.sqrt(np.mean((g_on_ref_grid - ref_data['g'])**2))
        f_l2 = np.sqrt(np.mean((f_on_ref_grid - ref_data['f'])**2))
        g_max_err = np.max(np.abs(g_rel_err))
        f_max_err = np.max(np.abs(f_rel_err))
        print(f"\n  对比指标:")
        print(f"    G 分量 L²误差: {g_l2:.4e}")
        print(f"    F 分量 L²误差: {f_l2:.4e}")
        print(f"    G 最大相对误差: {g_max_err:.2%}")
        print(f"    F 最大相对误差: {f_max_err:.2%}")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  对比图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ═══════════════════════════════════════════════════════════════
#   数值诊断工具
# ═══════════════════════════════════════════════════════════════

def compute_diagnostics(g, f, r, dr=DR):
    """
    计算波函数的一组数值诊断指标。

    参数:
        g, f: (N,) 波函数分量
        r: (N,) 径向坐标
        dr: float, 步长
    返回:
        diagnostics: dict 包含各种诊断指标
    """
    # 归一化
    norm_sq = np.trapz(g**2 + f**2, dx=dr)
    
    # 均方根半径
    r2_avg = np.trapz(r**2 * (g**2 + f**2), dx=dr) / norm_sq if norm_sq > 0 else 0
    rms_r = np.sqrt(max(0, r2_avg))
    
    # G 和 F 的相对强度
    g_power = np.trapz(g**2, dx=dr)
    f_power = np.trapz(f**2, dx=dr)
    ratio_fg = np.sqrt(f_power / g_power) if g_power > 1e-30 else float('inf')
    
    # 峰位置
    peak_idx = np.argmax(np.abs(g))
    r_peak = r[peak_idx]
    g_peak_val = g[peak_idx]
    
    # 尾部衰减率 (最后20%区域拟合)
    tail_start = int(0.8 * len(r))
    tail_g = np.abs(g[tail_start:])
    if np.any(tail_g > 1e-10):
        log_tail = np.log(tail_g[tail_g > 1e-10])
        r_tail = r[tail_start:][tail_g > 1e-10]
        if len(r_tail) > 2:
            fit_coeffs = np.polyfit(r_tail, log_tail, 1)
            decay_rate = -fit_coeffs[0]  # exp(-κr) 中的 κ
        else:
            decay_rate = 0.0
    else:
        decay_rate = 0.0

    # 正交性检验占位符 (需要另一个态才能算)
    
    return {
        'norm_integral': norm_sq,
        'rms_radius_fm': rms_r,
        'f_over_g_ratio': ratio_fg,
        'peak_position_fm': r_peak,
        'peak_value': g_peak_val,
        'tail_decay_rate_fm1': decay_rate,
        'g_integral': g_power,
        'f_integral': f_power,
    }


def print_diagnostics(diagnostics, label='PINN'):
    """格式打印诊断结果"""
    print(f"\n  [{label}] 数值诊断:")
    print(f"    归一化:       {diagnostics['norm_integral']:.6f} (目标=1.000000)")
    print(f"    RMS半径:      {diagnostics['rms_radius_fm']:.3f} fm")
    print(f"    |F|/|G| 比:   {diagnostics['f_over_g_ratio']:.3f}")
    print(f"    G峰位置:      {diagnostics['peak_position_fm']:.2f} fm (值={diagnostics['peak_value']:.4f})")
    print(f"    尾部衰减率:   {diagnostics['tail_decay_rate_fm1']:.3f} fm⁻¹")


def compare_with_fortran(pinn_result, ref_data, tolerance_energy=0.01, tolerance_waveform=0.05):
    """
    系统性比较 PINN 结果和 Fortran 参考解，返回是否通过验证。

    参数:
        pinn_result: dict from MVPSolver.evaluate()
        ref_data: dict from load_pkal_wavefunction()
        tolerance_energy: float, 能量允许的相对误差 (默认1%)
        tolerance_waveform: float, 波形L²允许的相对误差 (默认5%)

    返回:
        passed: bool, 是否全部通过
        report: dict, 详细报告
    """
    report = {'passed': True, 'details': {}}

    # 能量对比
    E_pinn = pinn_result['energy']
    E_ref = ref_data.get('energy', None)  # 如果参考数据包含能量
    
    if E_ref is not None:
        energy_err = abs(E_pinn - E_ref) / abs(E_ref) if E_ref != 0 else abs(E_pinn)
        passed_e = energy_err < tolerance_energy
        report['details']['energy'] = {
            'pinn': E_pinn, 'ref': E_ref,
            'rel_error_pct': energy_err * 100,
            'tolerance_pct': tolerance_energy * 100,
            'passed': passed_e,
        }
        if not passed_e:
            report['passed'] = False

    # 波形对比
    r_ref = ref_data['r']
    g_ref = ref_data['g']
    f_ref = ref_data['f']

    # 插值到同一网格
    from scipy.interpolate import interp1d
    r_common = r_ref  # 使用参考解的网格
    
    g_pinn_interp = interp1d(pinn_result['r'], pinn_result['g'], kind='linear', fill_value='extrapolate')(r_common)
    f_pinn_interp = interp1d(pinn_result['r'], pinn_result['f'], kind='linear', fill_value='extrapolate')(r_common)

    # L² 误差 (相对于参考解的范数)
    g_ref_norm = np.sqrt(np.trapz(g_ref**2, dx=DR))
    f_ref_norm = np.sqrt(np.trapz(f_ref**2, dx=DR))
    
    g_l2_err = np.sqrt(np.trapz((g_pinn_interp - g_ref)**2, dx=DR)) / (g_ref_norm + 1e-10)
    f_l2_err = np.sqrt(np.trapz((f_pinn_interp - f_ref)**2, dx=DR)) / (f_ref_norm + 1e-10)
    
    passed_g = g_l2_err < tolerance_waveform
    passed_f = f_l2_err < tolerance_waveform
    
    report['details']['waveform_G'] = {
        'L2_relative_error': g_l2_err,
        'tolerance': tolerance_waveform,
        'passed': passed_g,
    }
    report['details']['waveform_F'] = {
        'L2_relative_error': f_l2_err,
        'tolerance': tolerance_waveform,
        'passed': passed_f,
    }

    if not (passed_g and passed_f):
        report['passed'] = False

    # 归一化检验
    norm_pinn = pinn_result['norm_integral']
    passed_norm = abs(norm_pinn - 1.0) < 0.01
    report['details']['normalization'] = {
        'value': norm_pinn,
        'target': 1.0,
        'deviation': abs(norm_pinn - 1.0),
        'passed': passed_norm,
    }
    if not passed_norm:
        report['passed'] = False

    return report['passed'], report
