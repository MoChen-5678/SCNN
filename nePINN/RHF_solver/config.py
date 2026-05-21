#config.py
import os
import numpy as np
from .dirac_ws_solver import DiracWSSolver
from typing import Dict, List, Tuple, Optional
class RHFConfig:
    def __init__(
        self,
        A: int = 208,
        Z: int = 82,
        r_min: float = 0.001,
        r_max: float = 20.0,
        N_grid: int = 501,
        ECUT_p: float = 5.0,
        ECUT_n: float = 5.0,
        lambda_p: float = -7.0,
        lambda_n: float = -8.0,
    ):
        """
        初始化波函数计算器
        
        参数:
            A: 质量数
            Z: 电荷数
            ECUT_p, ECUT_n: 质子/中子单粒子能量相对化学势的截断(MeV)
            lambda_p, lambda_n: 质子/中子化学势(MeV)
        """
        self.A = A
        self.Z = Z
        self.kappa_list = [k for k in range(-10, 11) if k != 0]
        self.L_LETTERS = "spdfghijklmno"
        self.data = {
            "p": {"r": None, "states": {}},  # 使用字典存储，键为壳层标签
            "n": {"r": None, "states": {}},
        }
        self._calculated = False  # 标记是否已完成计算
        self.r_min = r_min
        self.r_max = r_max
        self.N_grid = N_grid

        # --------- 配对相关参数：能量截断和化学势 ----------
        # ECUT_tau: |ε_i - λ_tau| <= ECUT_tau 的单粒子态参与 BCS
        self.ECUT = {"p": ECUT_p, "n": ECUT_n}
        # λ_tau: 质子/中子的化学势 (MeV)
        self.lambdas = {"p": lambda_p, "n": lambda_n}

    def kappa_to_l_j2(self, kappa: int) -> Tuple[int, int]:
        """将kappa转换为(l, 2j)"""
        if kappa < 0:
            l = -kappa - 1
            j2 = 2 * l + 1
        else:
            l = kappa
            j2 = 2 * l - 1
        return l, j2

    def level_degeneracy(self, kappa: int) -> int:
        """
        根据 kappa 计算该单粒子轨道的简并度 g = 2j + 1
        """
        _, j2 = self.kappa_to_l_j2(kappa)  # j2 = 2j
        return j2 + 1

    def make_shell_label(self, kappa: int, nodes: int) -> str:
        """生成壳层标记"""
        n_shell = nodes + 1
        l, j2 = self.kappa_to_l_j2(kappa)
        l_letter = self.L_LETTERS[l]
        return f"{n_shell}{l_letter}{j2}/2"


    def calculate(self) -> None:
        """计算所有波函数"""
        if self._calculated:
            return

        for tau in ["p", "n"]:
            for kappa in self.kappa_list:
                solver = DiracWSSolver(A=self.A, Z=self.Z, nucleon_type=tau, kappa=kappa, r_min=self.r_min, r_max=self.r_max, N_grid=self.N_grid)
                eigvals, eigvecs = solver.solve_full()
                E_bound, idx_bound, nodes_list = solver.select_bound_states()

                if len(E_bound) == 0:
                    continue

                r = solver.r
                N_grid = len(r)

                if self.data[tau]["r"] is None:
                    self.data[tau]["r"] = r

                for E, idx, nodes in zip(E_bound, idx_bound, nodes_list):
                    shell_label = self.make_shell_label(kappa, nodes)
                    G = np.real(eigvecs[:N_grid, idx])
                    F = np.real(eigvecs[N_grid:2*N_grid, idx])

                    self.data[tau]["states"][shell_label] = {
                        "tau": tau,
                        "kappa": kappa,
                        "nodes": nodes,
                        "energy": E,
                        "G": G,
                        "F": F,
                    }

        self._calculated = True
    

    def get_occupations(
        self,
        tau: str,
        delta: float,
        use_cutoff: bool = True,) -> Dict[str, Dict[str, float]]:
        
        if not self._calculated:
            self.calculate()

        if tau not in ["p", "n"]:
            raise ValueError("tau 必须为 'p' 或 'n'")

        lam = self.lambdas[tau]
        ecut = self.ECUT[tau]

        states = self.data[tau]["states"]
        occupations: Dict[str, Dict[str, float]] = {}

        for shell_label, st in states.items():
            eps = float(st["energy"])       # 单粒子能量 ε_i
            kappa = int(st["kappa"])
            nodes = int(st["nodes"])

            # 与化学势的能量差
            d_eps = eps - lam

            # 能量截断：截断外的态不参加配对，给出“纯 HF”占据 v^2 = 0 或 1
            if use_cutoff and abs(d_eps) > ecut:
                # 纯 HF 占据：ε_i < λ => v^2=1; ε_i > λ => v^2=0
                if eps < lam:
                    v2 = 1.0
                else:
                    v2 = 0.0
                u2 = 1.0 - v2
                E_qp = abs(d_eps)  # 对应的准粒子能量（Δ=0 的极限）
            else:
                # 完整 BCS 占据几率(零温)
                #   E_i    = sqrt( (ε_i - λ)^2 + Δ^2 )
                #   v_i^2  = 1/2 * ( 1 - (ε_i - λ) / E_i )
                #   u_i^2  = 1/2 * ( 1 + (ε_i - λ) / E_i )
                E_qp = np.sqrt(d_eps**2 + delta**2)
                if E_qp == 0.0:
                    # 极端情况下避免 0/0
                    v2 = 0.5
                    u2 = 0.5
                else:
                    v2 = 0.5 * (1.0 - d_eps / E_qp)
                    u2 = 0.5 * (1.0 + d_eps / E_qp)

            g = self.level_degeneracy(kappa)     # 2j + 1
            N_level = g * v2                     # 该轨道实际占据的粒子数

            occupations[shell_label] = {
                "tau": tau,
                "energy": eps,
                "E_qp": E_qp,
                "lambda": lam,
                "delta": delta,
                "v2": v2,
                "u2": u2,
                "g": g,
                "N_level": N_level,
                "kappa": kappa,
                "nodes": nodes,
            }

        return occupations

    def get_wave_function(self, tau: str, shell_label: str, component: str = "G") -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定壳层的波函数
        
        参数:
            tau: 'p'(质子) 或 'n'(中子) 
            shell_label: 壳层标签，如 '1s1/2'
            component: 'G'(大分量) 或 'F'(小分量)
            
        返回:
            (r_grid, wave_function): 径向网格和波函数
        """
        if not self._calculated:
            self.calculate()

        if tau not in self.data or shell_label not in self.data[tau]["states"]:
            raise ValueError(f"未找到壳层 {shell_label} 的波函数")

        state = self.data[tau]["states"][shell_label]
        r = self.data[tau]["r"]
        wave_func = state[component.upper()]
        return r, wave_func

    def get_shell_info(self, tau: str, shell_label: str) -> Dict:
        """
        获取指定壳层的信息
        
        参数:
            tau: 'p'(质子) 或 'n'(中子)
            shell_label: 壳层标签
            
        返回:
            包含壳层信息的字典
        """
        if not self._calculated:
            self.calculate()

        if tau not in self.data or shell_label not in self.data[tau]["states"]:
            raise ValueError(f"未找到壳层 {shell_label} 的信息")

        state = self.data[tau]["states"][shell_label]
        return {
            "tau": state["tau"],
            "kappa": state["kappa"],
            "nodes": state["nodes"],
            "energy": state["energy"],
        }

    def list_shells(self, tau: str) -> List[str]:
        """
        列出所有可用的壳层
        
        参数:
            tau: 'p'(质子) 或 'n'(中子)
            
        返回:
            壳层标签列表
        """
        if not self._calculated:
            self.calculate()

        return list(self.data[tau]["states"].keys())

    def save_to_files(self, output_dir: str = "wavefunctions") -> None:
        """
        将波函数保存到文件
        
        参数:
            output_dir: 输出目录
        """
        if not self._calculated:
            self.calculate()

        os.makedirs(output_dir, exist_ok=True)

        for tau in ["p", "n"]:
            tau_label = "P" if tau == "p" else "N"
            r = self.data[tau]["r"]
            states = self.data[tau]["states"]

            # 按能量排序
            sorted_states = sorted(states.items(), key=lambda x: x[1]["energy"])

            # 写波函数文件
            for component in ["F", "G"]:
                filename = os.path.join(output_dir, f"Pb{self.A}.{component}-{tau_label}.dat")
                with open(filename, "w") as f:
                    # 写表头
                    header = "    r"
                    for shell_label, _ in sorted_states:
                        header += f"{tau_label}.{shell_label:>20s}"
                    header += "\n"
                    f.write(header)

                    # 写数据
                    for i_r, rv in enumerate(r):
                        line = f"{rv:8.4f}"
                        for _, state in sorted_states:
                            val = state[component][i_r]
                            line += f"{val:20.12E}"
                        line += "\n"
                        f.write(line)

            # 写能级谱表
            filename = os.path.join(output_dir, f"Pb{self.A}.spe-{tau_label}.dat")
            with open(filename, "w") as f:
                f.write("# tau  shell     kappa  nodes   Energy(MeV)\n")
                for shell_label, state in sorted_states:
                    f.write(
                        f"{tau_label:>3s}  "
                        f"{shell_label:<8s}  "
                        f"{state['kappa']:>5d}  "
                        f"{state['nodes']:>5d}  "
                        f"{state['energy']:>12.6f}\n"
                    )
