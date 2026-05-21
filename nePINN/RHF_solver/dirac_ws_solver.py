# dirac_ws_solver.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .ws_solver import WoodsSaxonSolver, Nucleus as WSNucleus, PotentialConfig, HBARC

class DiracWSSolver:
    def __init__(self, A=208, Z=82, nucleon_type='n', kappa=-1, 
                 r_min=0.001, r_max=20.0, N_grid=1001):
        """
        初始化Dirac方程Woods-Saxon势求解器
        
        参数:
            A: 核质量数
            Z: 核电荷数
            nucleon_type: 'p' 或 'n'，指定粒子类型
            kappa: Dirac量子数
            r_min: 径向坐标最小值 (fm)
            r_max: 径向坐标最大值 (fm)
            N_grid: 网格点数
        """
        self.A = A
        self.Z = Z
        self.N = A - Z
        self.nucleon_type = nucleon_type
        self.kappa = kappa
        self.r_min = r_min
        self.r_max = r_max
        self.N_grid = N_grid
        
        # 物理常数
        self.hbar_c = HBARC
        self.m_p = 938.272  # MeV
        self.m_n = 939.565  # MeV
        self.m = self.m_p if nucleon_type == 'p' else self.m_n
        
        # 初始化网格
        self.r = np.linspace(r_min, r_max, N_grid)
        self.dr = self.r[1] - self.r[0]
        
        # 初始化势场
        self._init_potential()
        
        # 初始化差分算子
        self._init_differential_operators()
        
        # 构建哈密顿量
        self._build_hamiltonian()
    
    def _init_potential(self):
        """初始化Woods-Saxon势场"""
        solver = WoodsSaxonSolver(A=self.A, Z=self.Z, dr=self.dr, r_max=self.r_max)
        it = 1 if self.nucleon_type == 'p' else 0
        
        # 获取势场数据并插值到当前网格
        VPS_ws = solver.gen.VPS[:, it]
        VMS_ws = solver.gen.VMS[:, it]
        r_ws = solver.gen.r
        
        VPS_interp = interp1d(r_ws, VPS_ws, kind='cubic', fill_value='extrapolate')
        VMS_interp = interp1d(r_ws, VMS_ws, kind='cubic', fill_value='extrapolate')
        
        self.VPS_diag = np.diag(VPS_interp(self.r))
        self.VMS_diag = np.diag(VMS_interp(self.r))
    
    def _init_differential_operators(self):
        """初始化差分算子"""
        if self.kappa < 0:
            self.D_G = self._create_D_Zhang2022(self.N_grid, self.dr, +1)
            self.D_F = self._create_D_Zhang2022(self.N_grid, self.dr, -1)
        else:
            self.D_G = self._create_D_Zhang2022(self.N_grid, self.dr, -1)
            self.D_F = self._create_D_Zhang2022(self.N_grid, self.dr, +1)
    
    def _create_D_Zhang2022(self, N, h, parity):
        """创建7点差分矩阵"""
        if N < 7:
            raise ValueError("N must be >= 7 for 7-point stencil.")
        
        D = np.zeros((N, N))
        coef = 1.0 / (60.0 * h)
        
        if parity == +1:
            for i in range(N - 6):
                D[i, i]     = -147.0 * coef
                D[i, i+1]   =  360.0 * coef
                D[i, i+2]   = -450.0 * coef
                D[i, i+3]   =  400.0 * coef
                D[i, i+4]   = -225.0 * coef
                D[i, i+5]   =   72.0 * coef
                D[i, i+6]   =  -10.0 * coef
            
            for i in range(N - 6, N):
                D[i, i]     =  147.0 * coef
                D[i, i-1]   = -360.0 * coef
                D[i, i-2]   =  450.0 * coef
                D[i, i-3]   = -400.0 * coef
                D[i, i-4]   =  225.0 * coef
                D[i, i-5]   =  -72.0 * coef
                D[i, i-6]   =   10.0 * coef
        else:
            for i in range(6, N):
                D[i, i]     =  147.0 * coef
                D[i, i-1]   = -360.0 * coef
                D[i, i-2]   =  450.0 * coef
                D[i, i-3]   = -400.0 * coef
                D[i, i-4]   =  225.0 * coef
                D[i, i-5]   =  -72.0 * coef
                D[i, i-6]   =   10.0 * coef
            
            for i in range(6):
                D[i, i]     = -147.0 * coef
                D[i, i+1]   =  360.0 * coef
                D[i, i+2]   = -450.0 * coef
                D[i, i+3]   =  400.0 * coef
                D[i, i+4]   = -225.0 * coef
                D[i, i+5]   =   72.0 * coef
                D[i, i+6]   =  -10.0 * coef
        
        return D
    
    def _build_hamiltonian(self):
        """构建哈密顿量矩阵"""
        K_diag = self.kappa / self.r
        K = np.diag(K_diag)
        W_single = np.zeros((self.N_grid, self.N_grid))
        
        self.H11 = self.VPS_diag + W_single
        self.H22 = self.VMS_diag + W_single
        self.H12 = -self.D_F + K
        self.H21 = self.D_G + K
        
        self.H_fm = np.block([
            [self.H11, self.H12],
            [self.H21, self.H22]
        ])
    
    def solve_full(self):
        """完整哈密顿量对角化"""
        eigvals_fm, eigvecs = np.linalg.eigh(self.H_fm)
        idx = np.argsort(eigvals_fm)
        self.eigvals_fm = eigvals_fm[idx]
        self.eigvecs = eigvecs[:, idx]
        self.eigvals_MeV = self.eigvals_fm * self.hbar_c
        return self.eigvals_MeV, self.eigvecs
    
    def solve_kinetic_balance(self):
        """动量平衡方法求解"""
        I = np.eye(self.N_grid)
        D = self.D_G if self.kappa < 0 else self.D_F
        B = (self.hbar_c / (2.0 * self.m)) * (D + np.diag(self.kappa/self.r))
        
        H_eff = self.H11 + self.H12 @ B + B.T @ self.H21 + B.T @ self.H22 @ B
        S_eff = I + B.T @ B
        
        Se_vals, Se_vecs = np.linalg.eigh(S_eff)
        Se_inv_sqrt = Se_vecs @ np.diag(Se_vals**-0.5) @ Se_vecs.T
        
        H_tilde = Se_inv_sqrt @ H_eff @ Se_inv_sqrt
        E_vals, Y = np.linalg.eigh(H_tilde)
        
        G_all = Se_inv_sqrt @ Y
        F_all = B @ G_all
        self.eigvecs = np.vstack([G_all, F_all])
        self.eigvals_fm = E_vals
        self.eigvals_MeV = self.eigvals_fm * self.hbar_c
        
        return self.eigvals_MeV, self.eigvecs
    
    def count_nodes(self, G, threshold=1e-4):
        """计算波函数节点数"""
        s = np.sign(G)
        s[np.abs(G) < threshold] = 0.0
        nodes = 0
        for i in range(1, len(s)):
            if s[i-1] * s[i] < 0:
                nodes += 1
        return nodes
    
    
    def select_bound_states(self, eps_min=-1.5*71.2, eps_max=10.0, 
                           tail_fraction=0.1, tail_tol=1e-3):
        """筛选束缚态"""
        N = self.N_grid
        n_tail = max(5, int(N * tail_fraction))
        r_tail = self.r[-n_tail:]
        
        idx_bound = []
        E_bound = []
        nodes_list = []  # 添加节点数列表
        
        for i, E in enumerate(self.eigvals_MeV):
            eps = E
            if not (eps_min < eps < eps_max):
                continue
                
            psi = self.eigvecs[:, i]
            if len(psi) == 2*N:
                G = psi[0:N]
                F = psi[N:2*N]
            else:
                G = psi
                D = self.D_G if self.kappa < 0 else self.D_F
                B = (self.hbar_c / (2.0 * self.m)) * (D + np.diag(self.kappa/self.r))
                F = B @ G
            
            norm = np.sqrt(np.trapz(G**2 + F**2, self.r))
            if norm == 0:
                continue
                
            G_norm = G / norm
            F_norm = F / norm
            
            G_tail = G_norm[-n_tail:]
            F_tail = F_norm[-n_tail:]
            tail_norm = np.sqrt(np.trapz(G_tail**2 + F_tail**2, r_tail))
            if tail_norm > tail_tol:
                continue
            
            nodes = self.count_nodes(G_norm)
            idx_bound.append((i, nodes, E))
            E_bound.append(E)
            nodes_list.append(nodes)  # 添加节点数到列表
        
        return np.array(E_bound), [i for (i, nodes, E) in idx_bound], nodes_list  # 返回节点数列表
    
    def plot_wavefunction(self, state_idx):
        """绘制指定态的波函数"""
        if state_idx >= len(self.eigvals_MeV):
            raise ValueError("State index out of range")
            
        psi = self.eigvecs[:, state_idx]
        N = self.N_grid
        
        if len(psi) == 2*N:
            G = psi[0:N]
            F = psi[N:2*N]
        else:
            G = psi
            D = self.D_G if self.kappa < 0 else self.D_F
            B = (self.hbar_c / (2.0 * self.m)) * (D + np.diag(self.kappa/self.r))
            F = B @ G
        
        norm = np.sqrt(np.trapz(G**2 + F**2, self.r))
        G = G / norm
        F = F / norm
        
        plt.figure()
        plt.plot(self.r, G, label='G(r)')
        plt.plot(self.r, F, label='F(r)')
        plt.xlabel("r (fm)")
        plt.ylabel("Wavefunction")
        plt.title(f"DWS-Dirac (matrix) using Woods-Saxon potential, κ={self.kappa}, type={self.nucleon_type}")
        plt.legend()
        plt.grid(True)
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建求解器实例
    solver = DiracWSSolver(
        A=208,          # 核质量数
        Z=82,           # 核电荷数
        nucleon_type='n',  # 粒子类型
        kappa=-1,       # Dirac量子数
        r_min=0.001,    # 最小半径
        r_max=20.0,     # 最大半径
        N_grid=1001     # 网格点数
    )
    
    # 选择求解方法
    print("请选择求解器:")
    print("1. 完整哈密顿量对角化")
    print("2. 动量平衡方法")
    choice = input("输入选择 (1 或 2): ")
    
    if choice == "1":
        eigvals_MeV, eigvecs = solver.solve_full()
    elif choice == "2":
        eigvals_MeV, eigvecs = solver.solve_kinetic_balance()
    else:
        print("无效选择，使用默认的完整哈密顿量对角化")
        eigvals_MeV, eigvecs = solver.solve_full()
    
    # 打印前10个本征值
    print("前 10 个本征值 (MeV):")
    for i in range(10):
        E = eigvals_MeV[i]
        print(f"  {i}: E = {E:.6f} MeV")
    
    # 筛选束缚态
    E_bound, idx_bound, nodes_list = solver.select_bound_states()
    if len(idx_bound) > 0:
        print("\n找到的束缚态:")
        for i, idx in enumerate(idx_bound):
            print(f"  态 {i}: E = {E_bound[i]:.6f} MeV")
        
        # 选择并绘制波函数
        i = input("\n选择要绘制的束缚态的索引号: ")
        try:
            i = int(i)
            if 0 <= i < len(idx_bound):
                solver.plot_wavefunction(idx_bound[i])
            else:
                print("索引超出范围")
        except ValueError:
            print("请输入有效的数字")
    else:
        print("没有找到束缚态")
