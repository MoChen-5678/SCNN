from re import M
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from mathtools import create_D,kappa_R,parse_shell_label,density_pm,simpson_weights,build_local_hamiltonian
from nonlocal_potential import build_nonlocal_potential
from config import RHFConfig
from density import compute_local_densities,compute_nonlocal_densities
from parameter import get_param_set
from hartree_potential import compute_hartree_potentials
from tensor import get_tensor_self_energy

#grid parameters
r_min = 0.1
r_max = 20.1
h = 201

#nuclear parameters
kappa = -1
A = 18
Z = 8
N = A-Z
tau = 'p'
mass = 938.272 if tau == 'p' else 939.565
#constant
p4 = 1/(np.pi*4)
param = get_param_set("PKA1")


D_G,D_F,r = create_D(r_min,r_max,h,kappa)
kapR = kappa_R(kappa,r)
weights = simpson_weights(r)
W_mat = np.diag(weights)
r2 = np.tile(r, 2)  # 将r数组重复两次，形成长度为2h的数组
config = RHFConfig(A=A,Z=Z,r_min=r_min,r_max=r_max,N_grid=h)
w = simpson_weights(r2)
# Calculate neutron density
tau_n = 'n'
occupations_n = config.get_occupations(tau=tau_n,delta=12/np.sqrt(A),use_cutoff=True)
pre_density_n = {}
for state_name, state_data in occupations_n.items():
    w_f_G = config.get_wave_function(tau=tau_n, shell_label=state_name, component='G')
    w_f_F = config.get_wave_function(tau=tau_n, shell_label=state_name, component='F')
    g_value = parse_shell_label(state_name)
    #Store the data in a dictionary
    pre_density_n[state_name] = {
        'v2': state_data['v2'],
        'g': g_value,
        'G': w_f_G[1],
        'F': w_f_F[1]
    }
rho_s_n,rho_v_n,rho_T_n = compute_local_densities(r=r,pre_density=pre_density_n)
Rpp_n,Rpm_n,Rmp_n,Rmm_n = compute_nonlocal_densities(pre_density=pre_density_n,N=h)

# Calculate proton density
tau_p = 'p'
occupations_p = config.get_occupations(tau=tau_p,delta=12/np.sqrt(A),use_cutoff=True)
pre_density_p = {}
for state_name, state_data in occupations_p.items():
    w_f_G = config.get_wave_function(tau=tau_p, shell_label=state_name, component='G')
    w_f_F = config.get_wave_function(tau=tau_p, shell_label=state_name, component='F')
    g_value = parse_shell_label(state_name)
    #Store the data in a dictionary
    pre_density_p[state_name] = {
        'v2': state_data['v2'],
        'g': g_value,
        'G': w_f_G[1],
        'F': w_f_F[1]
    }

#Density Calculation Summary
rho_s_p,rho_v_p,rho_T_p = compute_local_densities(r=r,pre_density=pre_density_p)
Rpp_p,Rpm_p,Rmp_p,Rmm_p = compute_nonlocal_densities(pre_density=pre_density_p,N=h)
rho_s,rho_v,rho_T,rs_3,rv_3,rt_3 = density_pm(rho_s_n,rho_v_n,rho_T_n,rho_s_p,rho_v_p,rho_T_p)
#Computing the self-energy of local tensors
sigma_T_iso = get_tensor_self_energy(r=r,rho_v_total=rho_v,rho_v_iso = rv_3,rho_t_iso=rt_3,param=param,weights=weights,tau=tau)
#Pre-calculation of nonlocal Hamiltonians
S_p,S_n,V_p,V_n,Sigma_plus_p,Sigma_minus_p,Sigma_plus_n,Sigma_minus_n = compute_hartree_potentials(r=r,rho_s_p=rho_s_p,rho_s_n=rho_s_n,rho_v_p=rho_v_p,rho_v_n=rho_v_n,param=param,weights=weights)
#Assemble Hamiltonian

# 只包含Hartree部分的哈密顿量
H_hartree_only = build_local_hamiltonian(kapR=kapR,mass=mass,r=r,Sigma_plus=Sigma_plus_p,Sigma_minus=Sigma_minus_p,DG=D_G,DF=D_F,Sigma_T =None)

# 完整哈密顿量(包含Fock)
H_hartree = build_local_hamiltonian(kapR=kapR,mass=mass,r=r,Sigma_plus=Sigma_plus_p,Sigma_minus=Sigma_minus_p,DG=D_G,DF=D_F,Sigma_T = sigma_T_iso)
H_fock = build_nonlocal_potential(r=r,param=param,weights=weights,target_kappa=kappa,target_tau=tau,occupations_n=occupations_n,pre_density_n=pre_density_n,occupations_p=occupations_p,pre_density_p=pre_density_p,rho_v=rho_v,rv_3=rv_3,rt_3=rt_3)


H_f = np.zeros((h+h, h+h))
H_f[0:h, 0:h] = H_fock['Vpp']  * (r[1]-r[0])/6 # Top-Left (F-F block): Vpp
# Bottom-Right (F-F block): Vmm
H_f[h:, h:]   = H_fock['Vmm']  * (r[1]-r[0])/6 
# Top-Right (G-F block): Vpm
H_f[0:h, h:]  = H_fock['Vpm']  * (r[1]-r[0])/6 
# Bottom-Left (F-G block): Vmp
H_f[h:, 0:h]  = H_fock['Vmp']  * (r[1]-r[0])/6 

# ==========================================
# 添加的对称性检查代码 (包含相对误差)
# ==========================================

print("\n" + "="*40)
print("开始检查哈密顿量矩阵的对称性")
print("="*40)

# 1. 构建总哈密顿量 (Hartree + Fock)
H_total = H_hartree + H_f

eigE,eigV =np.linalg.eigh(H_hartree_only)

print(eigE,f"Mev")