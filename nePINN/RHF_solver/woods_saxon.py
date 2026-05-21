#woods-saxon.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


# ===================== 基本数据结构 =====================

@dataclass
class Nucleus:
    """核参数：质量数 A，(N, Z)"""
    amas: float       # 质量数 A
    npr: np.ndarray   # [N, Z]


@dataclass
class ParticleSet:
    """粒子参数：amu[0]=中子质量(MeV), amu[1]=质子质量(MeV)"""
    amu: np.ndarray


@dataclass
class WaveB:
    """和 Fortran 里的 WAVEB 一样的波函数容器"""
    G: np.ndarray  # 大分量
    F: np.ndarray  # 小分量


@dataclass
class BoundState:
    """一个束缚态的全部信息"""
    E: float          # 能量 (MeV)
    kappa: int        # Dirac κ
    nodes: int        # 节点数
    r: np.ndarray     # 半径网格 (fm)
    G: np.ndarray     # 大分量
    F: np.ndarray     # 小分量


# ===================== 常数 & 工具函数 =====================

# Fortran 里常用的常数
one = 1.0
two = 2.0
half = 0.5
third = 1.0 / 3.0
zero = 0.0

# hbar*c，Fortran 里是 hbc
HBARC = 197.3269804  # MeV·fm


def simpson_integral(y: np.ndarray, h: float) -> float:
    """
    Simpson 求积，等间距网格.
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n % 2 == 0:
        # 必须奇数点，去掉最后一点
        n -= 1
        y = y[:n]
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:n-1:2]) + 2.0 * np.sum(y[2:n-2:2])
    return s * h / 3.0


def det2(a: np.ndarray) -> float:
    """2x2 行列式"""
    return a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]


def brent_bisect(fun, a: float, b: float, fa: float, fb: float,
                 tol: float = 1e-12, max_iter: int = 100) -> float:
    """
    简化版 Brent/bisection 根搜索，
    Fortran 里用的是 RTBRENT(DETB, ...).
    """
    # 简化起见，如果 fa, fb 同号就直接抛异常
    if fa * fb > 0:
        raise RuntimeError("Root not bracketed in [a,b]")

    left, right = a, b
    f_left, f_right = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = fun(mid)
        if abs(f_mid) < tol:
            return mid
        if f_left * f_mid < 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
        if abs(right - left) < tol:
            return 0.5 * (left + right)
    return 0.5 * (left + right)


# ===================== Woods–Saxon 势 =====================

class WoodsSaxonPotential:
    """
    Python 版的 WOODS 子程序：
    构造 Dirac 方程里用的 VPS(r), VMS(r)（Z 也带上库仑势）
    """

    def __init__(self,
                 pset: ParticleSet,
                 nuc: Nucleus,
                 dr: float,
                 r_max: float):
        self.pset = pset
        self.nuc = nuc
        self.dr = dr
        self.r_max = r_max

        # 网格
        npt = int(r_max / dr) + 1
        self.r = np.linspace(dr * 1e-2, r_max, npt)  # 避免 r=0
        self.npt = npt

        # 2 个 isospin: 0 = 中子, 1 = 质子
        self.VPS = np.zeros((npt, 2))
        self.VMS = np.zeros((npt, 2))

        # 这些参数直接抄 BASE.f90
        # V0, AKV, VSO(2), R0V(2), R0S(2), AV(2), AS(2)
        self.V0 = -71.28
        self.AKV = 0.4616
        self.VSO = np.array([11.1175, 8.9698])
        self.R0V = np.array([1.2334, 1.2496])
        self.R0S = np.array([1.1443, 1.1400])
        self.AV = np.array([0.615, 0.6124])
        self.AS = np.array([0.6476, 0.6469])

        self.build_potentials()

    def build_potentials(self):
        """
        模仿 Fortran WOODS 的逻辑，构造 VPS, VMS.
        """
        A = self.nuc.amas
        N = self.nuc.npr[0]
        Z = self.nuc.npr[1]

        # amu/hbc * 2
        EMCC = self.pset.amu / HBARC * two  # 2*m/(ħc)

        # 半径
        r = self.r

        for it in range(2):  # 0:中子 1:质子
            ita = 1 - it  # 另外一种
            RAV = self.R0V[it] * A ** (1.0 / 3.0)
            RAS = self.R0S[it] * A ** (1.0 / 3.0)

            # VP 是等效 (V+S)/2 之类的量，按 BASE 的写法
            VP = self.V0 * (1.0 - self.AKV * (self.nuc.npr[it] - self.nuc.npr[ita]) / A) / HBARC
            VLS = VP * self.VSO[it]  # 自旋轨道强度，也按 BASE 抄

            # Woods–Saxon 形式
            argV = (r - RAV) / self.AV[it]
            argS = (r - RAS) / self.AS[it]

            VPS = np.zeros_like(r)
            VMS = np.zeros_like(r)

            maskV = argV <= 65.0
            VPS[maskV] = VP / (1.0 + np.exp(argV[maskV]))
            # 大于 65 时势视为 0

            maskS = argS <= 65.0
            VMS[maskS] = -VLS / (1.0 + np.exp(argS[maskS]))

            # 质量项
            VMS -= EMCC[it]

            # 库仑势（只对质子）
            if it == 1 and Z > 0:
                # 非严格照搬 Fortran 的库仑写法，这里用简单点形式：
                # Vc(r) = 3Ze^2/(2R) - Ze^2 r^2/(2R^3) (r<R), = Ze^2/r (r>=R)
                alpha = 1.0 / 137.035999  # 精细结构常数
                Rch = 1.2 * A ** (1.0 / 3.0)
                C = alpha * Z
                Vc = np.empty_like(r)
                inside = r < Rch
                outside = ~inside
                Vc[inside] = C * (3.0 - (r[inside] / Rch) ** 2) / (2.0 * Rch)
                Vc[outside] = C / r[outside]
                VPS += Vc / HBARC
                VMS += Vc / HBARC

            self.VPS[:, it] = VPS
            self.VMS[:, it] = VMS


# ===================== Dirac + 匹配法 (DETB + MATCH) =====================

class DiracWoodsSaxonSolver:
    """
    等价于 BASE.f90 模块里 DETB + MATCH 的 Python 实现。
    在给定 κ、isospin 下，用匹配法找能级 + 波函数。
    """

    def __init__(self,
                 pot: WoodsSaxonPotential,
                 it: int,
                 kappa: int,
                 ien: int = 1):
        """
        it: 0 = 中子, 1 = 质子
        kappa: Dirac κ
        ien: 1 -> 正能束缚区(通常), -1 -> 负能区
        """
        self.pot = pot
        self.it = it
        self.kappa = kappa
        self.ien = ien

        self.r = pot.r
        self.npt = pot.npt
        self.h = pot.dr
        self.h2 = 0.5 * pot.dr

        self.VPS = pot.VPS[:, it]
        self.VMS = pot.VMS[:, it]

        # Fortran 里有 VPSH, VMSH (半格点值)，我们简单用平均近似
        self.VPSH = 0.5 * (self.VPS[:-1] + self.VPS[1:])
        self.VMSH = 0.5 * (self.VMS[:-1] + self.VMS[1:])

        # 匹配点：中间某个点，这里选在网格中点附近
        self.mat = self.npt // 2

        # SG, SF = G,F 的辅助数组（匹配前）
        self.SG = np.zeros(self.npt)
        self.SF = np.zeros(self.npt)
        # 匹配矩阵 GFM
        self.GFM = np.zeros((2, 2))

    # ---------- DETB: 构造匹配行列式 ----------

    def detb(self, E: float) -> float:
        """
        等价 Fortran 的 FUNCTION DETB(EIG)：
        给定能量 E(MeV)，往里+往外积分 Dirac 径向方程，
        在匹配点处构造 2x2 矩阵 GFM，并返回其行列式。
        """
        h = self.h
        h2 = self.h2
        npt = self.npt
        mat = self.mat
        kappa = self.kappa
        it = self.it

        VPS = self.VPS
        VMS = self.VMS
        VPSH = self.VPSH
        VMSH = self.VMSH

        SG = self.SG
        SF = self.SF
        GFM = self.GFM

        # 把 MeV 的 E 变成和 Fortran 一样的 EIGFM = E/hbc
        EIGFM = E / HBARC

        # ---- r=0 端的初始条件 (对应 Fortran 的 ORIGIN 部分) ----
        origin = 1.0e-5
        SG[0] = 0.0
        SF[0] = 0.0

        # Fortran 是从 RXM(2) 开始做正规化，这里用 r[1]
        if kappa > 0:
            alph = (2.0 * kappa + 1.0) / h * origin
            beta = EIGFM - VMS[1]
        else:
            alph = (EIGFM - VPS[1]) * origin
            beta = (2.0 * kappa - 1.0) / h

        SG[1] = origin
        SF[1] = alph / beta

        # ---- r->∞ 端的初值 ----
        if self.ien == 1:
            SG[-1] = 0.0
            SF[-1] = 1.0
        else:
            SG[-1] = 1.0
            SF[-1] = 0.0

        # ---- 向内积分 (NPT -> MAT) ----
        AG = np.zeros(4)
        AF = np.zeros(4)

        jtem = npt - mat
        for j in range(1, jtem + 1):
            i = npt - j
            x = self.r[i]
            x1 = kappa / x
            x2 = kappa / (x - h2)
            x4 = kappa / (x - h)

            sg1 = SG[i]
            sf1 = SF[i]

            for jk in range(4):
                if jk == 0:
                    sg2 = sg1
                    sf2 = sf1
                    u1g = x1
                    u1f = EIGFM - VMS[i]
                    u2f = x1
                    u2g = EIGFM - VPS[i]
                elif jk in (1, 2):
                    sg2 = sg1 + AG[jk - 1]
                    sf2 = sf1 + AF[jk - 1]
                    u1g = x2
                    u1f = EIGFM - VMSH[i - 1]
                    u2f = x2
                    u2g = EIGFM - VPSH[i - 1]
                else:  # jk == 3
                    sg2 = sg1 + 2.0 * AG[2]
                    sf2 = sf1 + 2.0 * AF[2]
                    u1g = x4
                    u1f = EIGFM - VMS[i - 1]
                    u2f = x4
                    u2g = EIGFM - VPS[i - 1]

                # AG, AF 是RK4增量
                AG[jk] = -h2 * (-u1g * sg2 + u1f * sf2)
                AF[jk] = -h2 * (u2f * sf2 - u2g * sg2)

            sg2 = (AG[0] + 2.0 * (AG[1] + AG[2]) + AG[3]) * third
            sf2 = (AF[0] + 2.0 * (AF[1] + AF[2]) + AF[3]) * third

            SG[i - 1] = sg1 + sg2
            SF[i - 1] = sf1 + sf2

        GFM[0, 0] = SG[mat]
        GFM[1, 0] = SF[mat]

        # ---- 向外积分 (2 -> MAT) ----
        AG[:] = 0.0
        AF[:] = 0.0

        item = mat - 1
        for i in range(1, item + 1):
            x = self.r[i]
            x1 = kappa / x
            x2 = kappa / (x + h2)
            x4 = kappa / (x + h)

            sg1 = SG[i]
            sf1 = SF[i]

            for jk in range(4):
                if jk == 0:
                    sg2 = sg1
                    sf2 = sf1
                    u1g = x1
                    u1f = EIGFM - VMS[i]
                    u2f = x1
                    u2g = EIGFM - VPS[i]
                elif jk in (1, 2):
                    sg2 = sg1 + AG[jk - 1]
                    sf2 = sf1 + AF[jk - 1]
                    u1g = x2
                    u1f = EIGFM - VMSH[i]
                    u2f = x2
                    u2g = EIGFM - VPSH[i]
                else:  # jk == 3
                    sg2 = sg1 + 2.0 * AG[2]
                    sf2 = sf1 + 2.0 * AF[2]
                    u1g = x4
                    u1f = EIGFM - VMS[i + 1]
                    u2f = x4
                    u2g = EIGFM - VPS[i + 1]

                AG[jk] = h2 * (-u1g * sg2 + u1f * sf2)
                AF[jk] = h2 * (u2f * sf2 - u2g * sg2)

            sg2 = (AG[0] + 2.0 * (AG[1] + AG[2]) + AG[3]) * third
            sf2 = (AF[0] + 2.0 * (AF[1] + AF[2]) + AF[3]) * third

            SG[i + 1] = sg1 + sg2
            SF[i + 1] = sf1 + sf2

        GFM[0, 1] = SG[mat]
        GFM[1, 1] = SF[mat]

        # ---- 匹配行列式 ----
        det_val = det2(GFM)
        return det_val

    # ---------- MATCH: 构造完整波函数 + 归一化 + 数节点 ----------

    def match(self, E: float) -> Tuple[WaveB, int, bool]:
        """
        等价于 Fortran SUBROUTINE MATCH(WAVE, NC, LPRX)
        在 detb(E)=0 的前提下，返回波函数、节点数及是否靠近匹配点的问题(LPRX)
        """
        # 先用 detb(E) 把 SG, SF, GFM 填好
        _ = self.detb(E)

        SG = self.SG.copy()
        SF = self.SF.copy()
        GFM = self.GFM.copy()

        npt = self.npt
        mat = self.mat
        h = self.h

        # 规范化 GFM 的两列，并同时缩放 SG, SF
        for col in range(2):
            s = np.sqrt(GFM[0, col] ** 2 + GFM[1, col] ** 2)
            if s == 0:
                continue
            s_inv = 1.0 / s
            GFM[:, col] *= s_inv

            if col == 0:
                i1, i2 = mat + 1, npt
            else:
                i1, i2 = 1, mat
            SG[i1 - 1:i2] *= s_inv
            SF[i1 - 1:i2] *= s_inv

        # 计算 MT * M (2x2 矩阵)
        AA = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                s = 0.0
                for i1 in range(2):
                    s += GFM[i1, i] * GFM[i1, j]
                AA[i, j] = s

        # 2x2 实对称矩阵的特征值/特征向量
        eigvals, eigvecs = np.linalg.eigh(AA)

        # 取第一根本征向量（按 BASE 的写法）
        v = eigvecs[:, 0]

        # 根据特征向量重构波函数
        G = np.zeros(npt)
        F = np.zeros(npt)

        # Fortran 中是 WAVE%G(i)=-SG(i)*AA(1,1) (i>MAT), +SG(i)*AA(2,1) (i<=MAT)
        # 这里等价用 v[0], v[1] 代替
        for i in range(mat, npt):
            G[i] = -SG[i] * v[0]
            F[i] = -SF[i] * v[0]
        for i in range(0, mat):
            G[i] = SG[i] * v[1]
            F[i] = SF[i] * v[1]

        # 归一化
        fun = G ** 2 + F ** 2
        norm = np.sqrt(simpson_integral(fun, h))
        if norm == 0:
            norm = 1.0
        G /= norm
        F /= norm

        # 数节点（看 IEN 大分量还是小分量）
        inode = 0
        lprx = False
        if self.ien == 1:
            comp = G
        else:
            comp = F

        for i in range(2, npt):
            if comp[i] * comp[i - 1] < 0:
                inode += 1
                if abs(i - mat) <= 1:
                    lprx = True

        return WaveB(G=G, F=F), inode, lprx

    # ---------- 在一个能量区间里扫描束缚态 ----------

    def find_bound_states(self,
                          e_min: float,
                          e_max: float,
                          n_max: int,
                          e_step: float = 1.0) -> List[BoundState]:
        """
        在 [e_min, e_max] 里扫描 detb(E) 的零点，找到前 n_max 个束缚态。
        e_min, e_max 单位 MeV
        """
        states: List[BoundState] = []

        E1 = e_min
        F1 = self.detb(E1)

        while E1 < e_max and len(states) < n_max:
            E2 = E1 + e_step
            if E2 > e_max:
                E2 = e_max
            F2 = self.detb(E2)

            # 找到变号 -> 存在本征值
            if F1 * F2 < 0.0:
                root = brent_bisect(self.detb, E1, E2, F1, F2, tol=1e-12)
                wave, nodes, lprx = self.match(root)

                bs = BoundState(
                    E=root,
                    kappa=self.kappa,
                    nodes=nodes,
                    r=self.r.copy(),
                    G=wave.G.copy(),
                    F=wave.F.copy()
                )
                states.append(bs)

            E1, F1 = E2, F2

        return states



import matplotlib.pyplot as plt

def plot_wavefunction(state: BoundState, title: str = None):
    """
    绘制束缚态的波函数
    
    Args:
        state: BoundState对象，包含波函数信息
        title: 可选的图表标题
    """
    plt.figure()
    
    # 绘制大分量和小分量
    plt.plot(state.r, state.G, label='Large component (G)', linewidth=2)
    plt.plot(state.r, state.F, label='Small component (F)', linewidth=2)
    
    # 添加标签和标题
    plt.xlabel('r (fm)', fontsize=12)
    plt.ylabel('Wave function', fontsize=12)
    if title is None:
        title = f'Bound State: κ={state.kappa}, E={state.E:.4f} MeV, nodes={state.nodes}'
    plt.title(title, fontsize=14)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 设置合适的坐标范围


    
    plt.tight_layout()
    plt.show()

# 在主程序中修改为：
if __name__ == "__main__":
    # 例子：用 BASE.f90 那套参数给 Pb-208 做一个 κ = -1 (1s1/2) 的基底束缚态
    A = 208
    Z = 82
    N = A - Z

    nuc = Nucleus(amas=A, npr=np.array([N, Z], dtype=float))
    # 粒子质量：近似中子/质子质量(MeV)
    pset = ParticleSet(amu=np.array([939.565, 938.272]))

    dr = 0.05   # fm
    r_max = 20.0  # fm

    pot = WoodsSaxonPotential(pset, nuc, dr, r_max)

    # κ = -1 -> j=1/2, l=0 (1s1/2)
    solver = DiracWoodsSaxonSolver(pot, it=1, kappa=-1, ien=1)

    # 扫描束缚能量区间，比如 -80 ~ 0 MeV
    bound_states = solver.find_bound_states(e_min=-80.0,
                                            e_max=-1.0,
                                            n_max=10,
                                            e_step=1.0)

    print(f"Found {len(bound_states)} bound states for kappa={solver.kappa}")
    for i, st in enumerate(bound_states):
        print(f"state #{i}: E = {st.E:.4f} MeV, nodes = {st.nodes}")
        # 绘制每个束缚态的波函数
        plot_wavefunction(st, title=f'O-16 Neutron State: κ={st.kappa}, E={st.E:.4f} MeV')

