# ws_solver.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# ===================== 1. 基础数据结构与配置 =====================

@dataclass
class Nucleus:
    """原子核参数"""
    A: float       # 质量数
    Z: int         # 质子数
    N: int         # 中子数

    @property
    def npr(self) -> np.ndarray:
        """返回 [N, Z] 数组，用于内部计算"""
        return np.array([self.N, self.Z], dtype=float)

@dataclass
class PotentialConfig:
    """
    Woods-Saxon 势场参数配置。
    默认值来自 standard parametrizations (如 Chepurnov 或类似全局拟合)。
    数组格式通常为 [中子参数, 质子参数]。
    """
    V0: float = -71.28          # 势阱深度 (MeV)
    AKV: float = 0.4616         # 同位旋依赖系数
    
    # 自旋轨道耦合强度 VSO (MeV) [中子, 质子]
    VSO: np.ndarray = field(default_factory=lambda: np.array([11.1175, 8.9698]))
    
    # 势阱半径参数 r0 (fm) [中子, 质子]
    R0V: np.ndarray = field(default_factory=lambda: np.array([1.2334, 1.2496]))
    
    # 自旋轨道半径参数 r0_ls (fm) [中子, 质子]
    R0S: np.ndarray = field(default_factory=lambda: np.array([1.1443, 1.1400]))
    
    # 势阱弥散度 a (fm) [中子, 质子]
    AV: np.ndarray = field(default_factory=lambda: np.array([0.615, 0.6124]))
    
    # 自旋轨道弥散度 a_ls (fm) [中子, 质子]
    AS: np.ndarray = field(default_factory=lambda: np.array([0.6476, 0.6469]))

@dataclass
class WaveFunction:
    """波函数结果容器"""
    r: np.ndarray     # 径向网格
    G: np.ndarray     # 大分量
    F: np.ndarray     # 小分量

@dataclass
class BoundState:
    """束缚态完整信息"""
    E: float          # 本征能量 (MeV)
    kappa: int        # Dirac 量子数
    l: int            # 轨道角动量 l
    j: float          # 总角动量 j
    nodes: int        # 径向节点数 (n)
    particle_type: str # 'neutron' or 'proton'
    wave: WaveFunction

    def __str__(self):
        """格式化输出：例如 1p3/2"""
        # 光谱符号转换
        spectro = "spdfghijk"[self.l] if self.l < 9 else "x"
        # 主量子数习惯上 = nodes + 1
        n_principal = self.nodes + 1
        return f"{self.particle_type[0].upper()}: {n_principal}{spectro}{int(2*self.j)}/2 (E={self.E:.4f} MeV)"

# ===================== 2. 物理常量与数学工具 =====================

HBARC = 197.3269804  # MeV·fm
NEUTRON_MASS = 939.565 # MeV
PROTON_MASS = 938.272  # MeV

def kappa_to_lj(kappa: int) -> Tuple[int, float]:
    """将 Dirac kappa 转换为 (l, j)"""
    j = abs(kappa) - 0.5
    if kappa < 0:
        l = abs(kappa) - 1  # j = l + 1/2
    else:
        l = kappa          # j = l - 1/2
    return int(l), j

def simpson_integral(y: np.ndarray, h: float) -> float:
    """Simpson 积分法"""
    n = len(y)
    if n < 2: return 0.0
    if n % 2 == 0: n -= 1
    s = y[0] + y[n-1] + 4.0 * np.sum(y[1:n-1:2]) + 2.0 * np.sum(y[2:n-2:2])
    return s * h / 3.0

def brent_bisect(fun, a: float, b: float, fa: float, fb: float, tol: float = 1e-10) -> float:
    """Brent/Bisection 混合求根"""
    if fa * fb > 0:
        raise ValueError("Root not bracketed")
    left, right = a, b
    for _ in range(100):
        mid = 0.5 * (left + right)
        f_mid = fun(mid)
        if abs(f_mid) < tol or abs(right - left) < tol:
            return mid
        if fa * f_mid < 0:
            right, fb = mid, f_mid
        else:
            left, fa = mid, f_mid
    return 0.5 * (left + right)

# ===================== 3. 核心物理引擎 =====================

class WoodsSaxonGenerator:
    """生成径向势 V(r) 和 S(r)"""
    def __init__(self, nucleus: Nucleus, config: PotentialConfig, dr: float, r_max: float):
        self.nuc = nucleus
        self.cfg = config
        self.dr = dr
        self.r = np.linspace(dr * 1e-2, r_max, int(r_max / dr) + 1)
        
        # 预计算势场
        self.VPS = np.zeros((len(self.r), 2)) # [r, isospin]
        self.VMS = np.zeros((len(self.r), 2))
        self._build_potentials()

    def _build_potentials(self):
        A = self.nuc.A
        r = self.r
        N_Z_arr = self.nuc.npr # [N, Z]
        
        masses = np.array([NEUTRON_MASS, PROTON_MASS])
        emcc = masses / HBARC * 2.0 # 2m/hbar*c

        for it in range(2): # 0=Neutron, 1=Proton
            # 基础参数提取
            R_V = self.cfg.R0V[it] * A**(1.0/3.0)
            R_S = self.cfg.R0S[it] * A**(1.0/3.0)
            a_V = self.cfg.AV[it]
            a_S = self.cfg.AS[it]
            
            # 同位旋不对称项
            ita = 1 - it
            vol_term = self.cfg.V0 * (1.0 - self.cfg.AKV * (N_Z_arr[it] - N_Z_arr[ita]) / A) / HBARC
            
            # Woods-Saxon 形状
            ws_func_v = 1.0 / (1.0 + np.exp((r - R_V) / a_V))
            # 自旋轨道项通常是势的导数形式，这里代码沿用原逻辑，直接用 WS 形状参数化 VLS
            # 注意：标准 WS 自旋轨道项通常是 Thomas 形式 (1/r dV/dr)，但此处沿用 Fortran 原代码的唯象写法
            ws_func_s = 1.0 / (1.0 + np.exp((r - R_S) / a_S))
            
            # 构造标量势与矢量势组合 (VPS = V+S, VMS = V-S 类似概念)
            # 注意：这里沿用原代码定义 VPS, VMS
            vp_val = vol_term
            vls_val = vol_term * self.cfg.VSO[it]

            # 截断过远距离的指数计算
            mask_v = ((r - R_V) / a_V) <= 60.0
            mask_s = ((r - R_S) / a_S) <= 60.0
            
            vps_arr = np.zeros_like(r)
            vms_arr = np.zeros_like(r)
            
            vps_arr[mask_v] = vp_val * ws_func_v[mask_v]
            vms_arr[mask_s] = -vls_val * ws_func_s[mask_s] # Minus sign from original logic
            
            vms_arr -= emcc[it] # 减去质量项

            # 添加库仑势 (仅对质子)
            if it == 1 and self.nuc.Z > 0:
                alpha = 1.0 / 137.035999
                R_ch = 1.2 * A**(1.0/3.0)
                const_c = alpha * self.nuc.Z
                vc = np.zeros_like(r)
                
                mask_in = r < R_ch
                mask_out = ~mask_in
                
                vc[mask_in] = const_c * (3.0 - (r[mask_in]/R_ch)**2) / (2.0 * R_ch)
                vc[mask_out] = const_c / r[mask_out]
                
                vps_arr += vc / HBARC
                vms_arr += vc / HBARC
            
            self.VPS[:, it] = vps_arr
            self.VMS[:, it] = vms_arr

class DiracSolver:
    """Dirac 方程求解器 (Shooting Method)"""
    def __init__(self, generator: WoodsSaxonGenerator, particle_type: str):
        self.gen = generator
        self.it = 1 if particle_type.lower() == 'proton' else 0
        self.r = generator.r
        self.h = generator.dr
        
        self.VPS = generator.VPS[:, self.it]
        self.VMS = generator.VMS[:, self.it]
        
        # 半整点势能 (简单平均)
        self.VPSH = 0.5 * (self.VPS[:-1] + self.VPS[1:])
        self.VMSH = 0.5 * (self.VMS[:-1] + self.VMS[1:])
        
        self.npt = len(self.r)
        self.mat = self.npt // 2  # 匹配点
        
        # 预分配内存
        self.SG = np.zeros(self.npt)
        self.SF = np.zeros(self.npt)
        self.GFM = np.zeros((2, 2))

    def _integrate_inward(self, kappa: int, eig_fm: float):
        """从无穷远向内积分"""
        # 边界条件 r->inf
        # 对于束缚态 (ien=1), 大分量衰减
        self.SG[-1] = 0.0
        self.SF[-1] = 1.0
        
        AG, AF = np.zeros(4), np.zeros(4)
        h2 = 0.5 * self.h
        third = 1.0/3.0
        
        for j in range(1, self.npt - self.mat + 1):
            i = self.npt - j
            x = self.r[i]
            # 系数预计算
            k_x = kappa / x
            k_x_h2 = kappa / (x - h2)
            k_x_h = kappa / (x - self.h)
            
            sg1, sf1 = self.SG[i], self.SF[i]
            
            # Runge-Kutta 4
            for step in range(4):
                if step == 0:
                    sg, sf = sg1, sf1
                    u1g, u2f = k_x, k_x
                    u1f = eig_fm - self.VMS[i]
                    u2g = eig_fm - self.VPS[i]
                elif step in (1, 2):
                    sg = sg1 + AG[step-1]
                    sf = sf1 + AF[step-1]
                    u1g, u2f = k_x_h2, k_x_h2
                    u1f = eig_fm - self.VMSH[i-1]
                    u2g = eig_fm - self.VPSH[i-1]
                else:
                    sg = sg1 + 2*AG[2]
                    sf = sf1 + 2*AF[2]
                    u1g, u2f = k_x_h, k_x_h
                    u1f = eig_fm - self.VMS[i-1]
                    u2g = eig_fm - self.VPS[i-1]
                
                fact = -h2 if step > 0 else -h2 # inward needs negative step effectively
                # 注意：原代码 inward 逻辑比较特殊，这里完全复刻原代码的代数结构
                AG[step] = -h2 * (-u1g * sg + u1f * sf)
                AF[step] = -h2 * (u2f * sf - u2g * sg)

            self.SG[i-1] = sg1 + (AG[0] + 2*(AG[1]+AG[2]) + AG[3]) * third
            self.SF[i-1] = sf1 + (AF[0] + 2*(AF[1]+AF[2]) + AF[3]) * third
            
        self.GFM[0, 0] = self.SG[self.mat]
        self.GFM[1, 0] = self.SF[self.mat]

    def _integrate_outward(self, kappa: int, eig_fm: float):
        """从原点向外积分"""
        origin = 1.0e-5
        self.SG[0] = 0.0
        self.SF[0] = 0.0
        
        # r -> 0 行为
        if kappa > 0:
            alph = (2.0 * kappa + 1.0) / self.h * origin
            beta = eig_fm - self.VMS[1]
            self.SG[1] = origin
            self.SF[1] = alph / beta if beta != 0 else 0
        else:
            alph = (eig_fm - self.VPS[1]) * origin
            beta = (2.0 * kappa - 1.0) / self.h
            self.SG[1] = origin
            self.SF[1] = alph / beta if beta != 0 else 0

        AG, AF = np.zeros(4), np.zeros(4)
        h2 = 0.5 * self.h
        third = 1.0/3.0

        for i in range(1, self.mat):
            x = self.r[i]
            k_x = kappa / x
            k_x_h2 = kappa / (x + h2)
            k_x_h = kappa / (x + self.h)
            
            sg1, sf1 = self.SG[i], self.SF[i]
            
            for step in range(4):
                if step == 0:
                    sg, sf = sg1, sf1
                    u1g, u2f = k_x, k_x
                    u1f = eig_fm - self.VMS[i]
                    u2g = eig_fm - self.VPS[i]
                elif step in (1, 2):
                    sg = sg1 + AG[step-1]
                    sf = sf1 + AF[step-1]
                    u1g, u2f = k_x_h2, k_x_h2
                    u1f = eig_fm - self.VMSH[i]
                    u2g = eig_fm - self.VPSH[i]
                else:
                    sg = sg1 + 2*AG[2]
                    sf = sf1 + 2*AF[2]
                    u1g, u2f = k_x_h, k_x_h
                    u1f = eig_fm - self.VMS[i+1]
                    u2g = eig_fm - self.VPS[i+1]
                
                AG[step] = h2 * (-u1g * sg + u1f * sf)
                AF[step] = h2 * (u2f * sf - u2g * sg)

            self.SG[i+1] = sg1 + (AG[0] + 2*(AG[1]+AG[2]) + AG[3]) * third
            self.SF[i+1] = sf1 + (AF[0] + 2*(AF[1]+AF[2]) + AF[3]) * third

        self.GFM[0, 1] = self.SG[self.mat]
        self.GFM[1, 1] = self.SF[self.mat]

    def calculate_determinant(self, E: float, kappa: int) -> float:
        """计算给定能量和 Kappa 下的匹配行列式"""
        eig_fm = E / HBARC
        self._integrate_inward(kappa, eig_fm)
        self._integrate_outward(kappa, eig_fm)
        # det = G_in * F_out - G_out * F_in
        return self.GFM[0, 0] * self.GFM[1, 1] - self.GFM[0, 1] * self.GFM[1, 0]

    def get_wavefunction(self, E: float, kappa: int) -> Tuple[WaveFunction, int]:
        """计算归一化波函数和节点数"""
        # 1. 确保 GFM 已经更新
        self.calculate_determinant(E, kappa)
        
        # 2. 匹配系数
        # 归一化 GFM 的两列以防止数值溢出
        for col in range(2):
            norm = np.hypot(self.GFM[0, col], self.GFM[1, col])
            if norm > 0:
                self.GFM[:, col] /= norm
                if col == 0: # inward
                    self.SG[self.mat:] /= norm
                    self.SF[self.mat:] /= norm
                else: # outward
                    self.SG[:self.mat+1] /= norm
                    self.SF[:self.mat+1] /= norm

        # 求解系数矩阵 M 的特征向量
        M = np.dot(self.GFM.T, self.GFM)
        _, eigvecs = np.linalg.eigh(M)
        vec = eigvecs[:, 0] # 最小特征值对应的向量 (对应行列式接近0)

        G_full = np.zeros(self.npt)
        F_full = np.zeros(self.npt)
        
        # 拼接
        # Outward (0 -> mat) * vec[1]
        G_full[:self.mat] = self.SG[:self.mat] * vec[1]
        F_full[:self.mat] = self.SF[:self.mat] * vec[1]
        # Inward (mat -> end) * -vec[0] (符号约定)
        G_full[self.mat:] = -self.SG[self.mat:] * vec[0]
        F_full[self.mat:] = -self.SF[self.mat:] * vec[0]
        
        # 3. 归一化
        rho = G_full**2 + F_full**2
        norm_fac = np.sqrt(simpson_integral(rho, self.h))
        if norm_fac > 0:
            G_full /= norm_fac
            F_full /= norm_fac
            
        # 4. 计算节点数 (主量 G)
        nodes = 0
        for i in range(2, self.npt):
            if G_full[i] * G_full[i-1] < 0:
                nodes += 1
        
        return WaveFunction(self.r, G_full, F_full), nodes

# ===================== 4. 对外接口 =====================

class WoodsSaxonSolver:
    """
    封装好的 Woods-Saxon 求解器。
    用法:
        solver = WoodsSaxonSolver(A=16, Z=8)
        states = solver.solve(particle='neutron', n_max=3)
    """
    def __init__(self, A: float, Z: int, 
                 dr: float = 0.05, r_max: float = 20.0,
                 config: PotentialConfig = None):
        self.nuc = Nucleus(A, Z, int(A-Z))
        self.config = config if config else PotentialConfig()
        self.gen = WoodsSaxonGenerator(self.nuc, self.config, dr, r_max)
        
    def solve(self, particle: str, kappa_list: List[int] = None, 
              e_range: Tuple[float, float] = (-80.0, -0.1), 
              e_step: float = 1.0) -> List[BoundState]:
        """
        求解特定粒子的束缚态。
        
        Args:
            particle: 'neutron' or 'proton'
            kappa_list: 要扫描的 kappa 值列表。默认扫描 s, p, d, f 轨道 (kappa = -1, 1, -2, 2, -3, 3, -4)
            e_range: 能量扫描范围 (MeV)
            e_step: 粗扫描步长 (MeV)
        """
        if kappa_list is None:
            kappa_list = [-1, 1, -2, 2, -3, 3, -4] # s1/2, p1/2, p3/2, d3/2, d5/2...
            
        solver_core = DiracSolver(self.gen, particle)
        results = []
        
        for kap in kappa_list:
            l, j = kappa_to_lj(kap)
            
            # 粗扫描寻找变号点
            e_curr = e_range[0]
            f_curr = solver_core.calculate_determinant(e_curr, kap)
            
            while e_curr < e_range[1]:
                e_next = e_curr + e_step
                if e_next > e_range[1]: e_next = e_range[1]
                
                f_next = solver_core.calculate_determinant(e_next, kap)
                
                # 发现根
                if f_curr * f_next < 0:
                    try:
                        # 精细求根
                        root_E = brent_bisect(
                            lambda e: solver_core.calculate_determinant(e, kap),
                            e_curr, e_next, f_curr, f_next
                        )
                        # 获取波函数
                        wave, nodes = solver_core.get_wavefunction(root_E, kap)
                        
                        st = BoundState(
                            E=root_E, kappa=kap, l=l, j=j, nodes=nodes,
                            particle_type=particle, wave=wave
                        )
                        results.append(st)
                    except ValueError:
                        pass # Bracket error, skip
                
                e_curr = e_next
                f_curr = f_next
        
        # 按能量排序
        results.sort(key=lambda x: x.E)
        return results

