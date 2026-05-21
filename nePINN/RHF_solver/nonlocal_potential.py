import numpy as np
from fock import (
    compute_fock_sigma_channel, 
    compute_fock_omega_channel,
    compute_fock_rhoV_channel, 
    compute_fock_rhoT_channel,
    compute_fock_rhoVT_channel, 
    compute_fock_pi_channel,
    compute_fock_coulomb_channel
)
from mathtools import parse_kappa_from_label

def build_nonlocal_potential(
    r: np.ndarray,
    param: dict,
    weights: np.ndarray,
    target_kappa: int,
    target_tau: str,
    occupations_n: dict,
    pre_density_n: dict,
    occupations_p: dict,
    pre_density_p: dict,
    rho_v: np.ndarray,
    rv_3: np.ndarray,
    rt_3: np.ndarray
) -> dict:
    
    N_grid = len(r)
    V_fock = {k: np.zeros((N_grid, N_grid)) for k in ['Vpp', 'Vmm', 'Vpm', 'Vmp']}
    zeros_mat = np.zeros((N_grid, N_grid))
    
    rho_v_total = rho_v
    target_tau_z = 1.0 if target_tau == 'n' else -1.0
    
    sources = [('n', pre_density_n), ('p', pre_density_p)]
    
    for source_tau, density_dict in sources:
        is_proton_source = (source_tau == 'p')
        source_tau_z = -1.0 if is_proton_source else 1.0
        
        for state_label, data in density_dict.items():
            if 'kappa' in data:
                kb = data['kappa']
            else:
                kb = parse_kappa_from_label(state_label)
            
            # 构建单壳层密度
            g_degen = data['g']
            v2_occ = data['v2']
            dens_factor = (g_degen * v2_occ) / (4.0 * np.pi)
            
            G = data['G']
            F = data['F']
            
            Rpp_b = dens_factor * np.outer(G, G)
            Rmm_b = dens_factor * np.outer(F, F)
            Rpm_b = dens_factor * np.outer(G, F)
            Rmp_b = dens_factor * np.outer(F, G)
            
            # --- 构造参数 ---
            # 为了避免 **args 传递多余参数导致报错，我们显式构建参数字典
            if is_proton_source:
                args = {
                    'Rpp_p': Rpp_b,     'Rmm_p': Rmm_b, 
                    'Rpp_n': zeros_mat, 'Rmm_n': zeros_mat,
                    'Rpm_p': Rpm_b,     'Rmp_p': Rmp_b,
                    'Rpm_n': zeros_mat, 'Rmp_n': zeros_mat
                }
            else:
                args = {
                    'Rpp_p': zeros_mat, 'Rmm_p': zeros_mat,
                    'Rpp_n': Rpp_b,     'Rmm_n': Rmm_b,
                    'Rpm_p': zeros_mat, 'Rmp_p': zeros_mat,
                    'Rpm_n': Rpm_b,     'Rmp_n': Rmp_b
                }

            # --- 计算通道 (显式传递参数，修复 TypeError) ---
            
            # 1. Sigma-S (仅需要 pp, mm)
            res_sig = compute_fock_sigma_channel(
                r, rho_v_total, 
                args['Rpp_p'], args['Rmm_p'], args['Rpp_n'], args['Rmm_n'],
                param=param, weights=weights, ka=target_kappa, kb=kb
            )
            V_fock['Vpp'] += res_sig['Vpp']
            V_fock['Vmm'] += res_sig['Vmm']
            
            # 2. Omega-V (仅需要 pp, mm)
            res_ome = compute_fock_omega_channel(
                r, rho_v_total, 
                args['Rpp_p'], args['Rmm_p'], args['Rpp_n'], args['Rmm_n'],
                param=param, weights=weights, ka=target_kappa, kb=kb
            )
            V_fock['Vpp'] += res_ome['Vpp']
            V_fock['Vmm'] += res_ome['Vmm']
            
            # 3. Rho-V (仅需要 pp, mm)
            res_rhoV = compute_fock_rhoV_channel(
                r, rho_v_total, 
                args['Rpp_p'], args['Rmm_p'], args['Rpp_n'], args['Rmm_n'],
                param=param, weights=weights, ka=target_kappa, kb=kb
            )
            V_fock['Vpp'] += target_tau_z * res_rhoV['Vpp_iso3']
            V_fock['Vmm'] += target_tau_z * res_rhoV['Vmm_iso3']
            
            # 4. Rho-T (仅需要 pm, mp)
            res_rhoT = compute_fock_rhoT_channel(
                r, rho_v_total, 
                rt_3, # Direct tensor density
                args['Rpm_p'], args['Rmp_p'], args['Rpm_n'], args['Rmp_n'],
                param, weights, target_kappa, kb, target_tau
            )
            V_fock['Vpm'] += target_tau_z * res_rhoT['Vpm_iso3']
            V_fock['Vmp'] += target_tau_z * res_rhoT['Vmp_iso3']
            
            # 5. Rho-VT (Mixing)
            # 构造 Isovector 密度 R_iso = R_b * source_tau_z
            Rpp_iso = Rpp_b * source_tau_z
            Rmm_iso = Rmm_b * source_tau_z
            Rpm_iso = Rpm_b * source_tau_z
            Rmp_iso = Rmp_b * source_tau_z
            
            res_rhoVT = compute_fock_rhoVT_channel(
                r, rho_v_total,
                rv_3, rt_3,  # Direct densities
                Rpp_iso, Rmm_iso, Rpm_iso, Rmp_iso,
                param, weights, target_kappa, kb, target_tau
            )
            
            V_fock['Vpp'] += target_tau_z * res_rhoVT['Vpp_iso3']
            V_fock['Vmm'] += target_tau_z * res_rhoVT['Vmm_iso3']
            V_fock['Vpm'] += target_tau_z * res_rhoVT['Vpm_iso3']
            V_fock['Vmp'] += target_tau_z * res_rhoVT['Vmp_iso3']
            
            # 6. Pi-PV (仅需要 pm, mp)
            res_pi = compute_fock_pi_channel(
                r, rho_v_total,
                rt_3,
                args['Rpm_p'], args['Rmp_p'], args['Rpm_n'], args['Rmp_n'],
                param, weights, target_kappa, kb
            )
            V_fock['Vpm'] += target_tau_z * res_pi['Vpm_iso3']
            V_fock['Vmp'] += target_tau_z * res_pi['Vmp_iso3']
            
            # 7. Coulomb (仅 pp, mm)
            if target_tau == 'p' and is_proton_source:
                res_coul = compute_fock_coulomb_channel(
                    r, args['Rpp_p'], args['Rmm_p'], 
                    param, weights, target_kappa, kb
                )
                V_fock['Vpp'] += res_coul['Vpp_p']
                V_fock['Vmm'] += res_coul['Vmm_p']

    return V_fock