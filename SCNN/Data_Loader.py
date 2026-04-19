import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 全局随机种子：确保多核素间划分一致
_SPLIT_SEED = 42

# ═══════════════════════════════════════════════════════════════
#   核素 → (Z, N) 映射表
#   用于宏观条件编码器 (FiLM) 的条件输入
# ═══════════════════════════════════════════════════════════════
ISOTOPE_ZN = {
    # 氧同位素链
    '14O': (8, 6), '16O': (8, 8), '18O': (8, 10), '20O': (8, 12),
    '22O': (8, 14), '24O': (8, 16),
    # 钙同位素链
    '36Ca': (20, 16), '38Ca': (20, 18), '40Ca': (20, 20),
    '42Ca': (20, 22), '44Ca': (20, 24), '46Ca': (20, 26), '48Ca': (20, 28),
    '50Ca': (20, 30), '52Ca': (20, 32),
    # 镍同位素链
    '56Ni': (28, 28), '58Ni': (28, 30), '60Ni': (28, 32), '62Ni': (28, 34),
    '64Ni': (28, 36), '68Ni': (28, 40), '72Ni': (28, 44), '78Ni': (28, 50),
    # 锡同位素链
    '100Sn': (50, 50), '112Sn': (50, 62), '116Sn': (50, 66),
    '120Sn': (50, 70), '124Sn': (50, 74), '132Sn': (50, 82),
    # 铅同位素链
    '204Pb': (82, 122), '206Pb': (82, 124), '208Pb': (82, 126), '210Pb': (82, 128),
    # 其他重要核素
    '86Kr': (36, 50), '88Sr': (38, 50), '90Zr': (40, 50), '92Mo': (42, 50),
}

# ★ 全核素列表（从数据目录自动扫描或手动指定）
ALL_ISOTOPES = list(ISOTOPE_ZN.keys())


def get_zn(isotope):
    """从核素名获取 (Z, N)，不在表中的尝试从名称解析"""
    if isotope in ISOTOPE_ZN:
        return ISOTOPE_ZN[isotope]
    # 尝试从名称解析: '208Pb' → A=208, 需要元素符号查Z
    try:
        m = re.match(r'(\d+)([A-Z][a-z]?)', isotope)
        if m:
            A = int(m.group(1))
            symbol = m.group(2)
            # 常见元素原子序数
            element_z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,
                         'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,
                         'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,
                         'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,
                         'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
                         'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,
                         'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
                         'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
                         'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83}
            Z = element_z.get(symbol, 0)
            if Z > 0:
                return (Z, A - Z)
    except Exception:
        pass
    # 兜底：返回 (0, 0)，训练时会用默认值
    print(f"  ⚠️ 核素 {isotope} 不在 ISOTOPE_ZN 映射表中，使用默认 (Z=0, N=0)")
    return (0, 0)


def build_datasets(data_dir, isotopes, max_seq_len=10, min_seq_len=3,
                   traj_usage_ratio=0.8, mode='train',
                   target_states=None, val_ratio=0.15, test_ratio=0.15):
    """
    按轨迹（state级别）构建 train/val/test 数据集。

    核心改进：
    ──────────────────────────────
    1. 自适应序列长度
    2. Y_target 为该 state 最终 loop 的收敛态波函数（而非下一步）
    3. 每个样本附加 (Z, N) 宏观量子数标量

    划分策略：每个核素内部按轨迹随机分层抽样，保证：
      1. 同一轨迹的所有样本不会跨集泄露
      2. 各集中 kappa 分布比例一致（分层采样）
      3. 多进程(DDP)间划分结果一致（固定种子）

    返回：
    ------
    ConcatDataset 或 None（如果该模式无样本）
    每个样本为七元组: (X, Y_11ch, kappa, is_proton, actual_len, z_num, n_num)
    """
    rng = np.random.default_rng(_SPLIT_SEED)

    # 第一遍：预扫描所有核素，收集轨迹元信息
    all_trajs = []

    for iso in isotopes:
        meta = _RHF_MetaDataset(
            data_dir=data_dir, isotope=iso,
            max_seq_len=max_seq_len, min_seq_len=min_seq_len,
            traj_usage_ratio=traj_usage_ratio, target_states=target_states
        )
        if len(meta.traj_meta) > 0:
            all_trajs.extend(meta.traj_meta)

    if not all_trajs:
        print("⚠️ 未找到任何有效轨迹！")
        return None

    total_n_samples = sum(t[2] for t in all_trajs)
    print(f"📦 数据集划分总览: {len(all_trajs)} 条轨迹, {total_n_samples} 个样本 "
          f"[max_seq={max_seq_len}, min={min_seq_len}, usage={traj_usage_ratio:.0%}]")

    # 按 kappa 分组后在各组内分层采样，保证分布一致
    kappa_groups = {}
    for t in all_trajs:
        k = t[3]
        if k not in kappa_groups:
            kappa_groups[k] = []
        kappa_groups[k].append(t)

    selected_trajs = {'train': [], 'val': [], 'test': []}
    for k, trajs in kappa_groups.items():
        rng.shuffle(trajs)
        n = len(trajs)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_test = max(0, n_test - 1)
            n_val = max(0, n_val - 1)
            n_train = n - n_val - n_test

        selected_trajs['train'].extend(trajs[:n_train])
        selected_trajs['val'].extend(trajs[n_train:n_train+n_val])
        selected_trajs['test'].extend(trajs[n_train+n_val:])

    # 构建目标模式的Dataset
    target_indices = selected_trajs.get(mode, [])
    datasets = []

    for iso in isotopes:
        iso_state_names = set()
        for t in target_indices:
            if t[0] == iso:
                iso_state_names.add(t[1])

        if not iso_state_names:
            continue

        ds = _RHF_Dataset(
            data_dir=data_dir, isotope=iso,
            max_seq_len=max_seq_len, min_seq_len=min_seq_len,
            traj_usage_ratio=traj_usage_ratio,
            target_states=target_states,
            allowed_states=iso_state_names
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        return None

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)

    print(f"   📊 Train={len(selected_trajs['train'])}轨迹 | Val={len(selected_trajs['val'])}轨迹 | Test={len(selected_trajs['test'])}轨迹")
    print(f"   ✅ [{mode}] 模式: {len(combined)} 个样本")

    return combined


class _RHF_MetaDataset:
    """预扫描阶段：仅收集轨迹元信息，不加载实际数据。用于 build_datasets 的分层划分"""

    def __init__(self, data_dir, isotope, max_seq_len=10, min_seq_len=3,
                 traj_usage_ratio=0.8, target_states=None):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.traj_usage_ratio = traj_usage_ratio
        self.target_states = target_states
        self.traj_meta = []  # [(isotope, state_name, n_samples, kappa), ...]

        wav_dir = os.path.join(data_dir, isotope, 'WAV')
        pot_dir = os.path.join(data_dir, isotope, 'POT')

        if not os.path.exists(wav_dir):
            return

        all_wav_files = sorted([f for f in os.listdir(wav_dir) if '.loop' in f])
        all_states = sorted(list(set(
            [re.match(r'(.*_state\d+)', f).group(1) for f in all_wav_files if re.match(r'(.*_state\d+)', f)]
        )))

        filtered_states = self._filter_states(all_states, all_wav_files, wav_dir)
        if not filtered_states:
            return

        # 收集每个态的元信息（含自适应seq_len和采样数估算）
        for state in filtered_states:
            state_files = sorted([f for f in all_wav_files if f.startswith(state)],
                                 key=lambda x: _extract_it_loop(x))

            # 计算有效点数
            valid_steps = []
            kappa = -1.0
            for wav_file in state_files:
                pot_file = wav_file.replace('.it', '_POT.it')
                wav_path = os.path.join(wav_dir, wav_file)
                pot_path = os.path.join(pot_dir, pot_file)
                if not os.path.exists(pot_path):
                    continue
                res = _parse_single_step(wav_path, pot_path, data_dir)
                if res is not None:
                    valid_steps.append(res[0])
                    kappa = res[1]

            L_total = len(valid_steps)
            if L_total < self.min_seq_len + 1:
                continue

            # 自适应序列长度
            eff_len = max(self.min_seq_len, min(self.max_seq_len, int(L_total * self.traj_usage_ratio)))

            # 样本数 = 滑动窗口数量
            n_windows = L_total - eff_len + 1
            stride = max(1, n_windows // max(n_windows, 1)) if n_windows > 0 else 1
            num_samples = (n_windows + stride - 1) // stride
            num_samples = max(1, num_samples)

            if num_samples > 0:
                self.traj_meta.append((isotope, state, num_samples, kappa))

    def _filter_states(self, all_states, all_wav_files, wav_dir):
        """目标态筛选（简化版：final即最终数据，不再做复杂收敛性判断）"""
        if self.target_states is None:
            return all_states

        def _parse_target_lj(s):
            m = re.match(r'(\d*)([a-z])(\d+)/2', s.lower())
            return (m.group(2), int(m.group(3)) / 2.0) if m else None

        target_ljs = [(ts, _parse_target_lj(ts)) for ts in self.target_states]
        filtered = []

        for state in all_states:
            state_files = [f for f in all_wav_files if f.startswith(state)]
            if not state_files:
                continue
            first_file = os.path.join(wav_dir, state_files[0])
            try:
                with open(first_file, 'r') as f:
                    content = f.read()
                match = re.search(r'State:\s*([NP])\.(\d+)([a-z])\.(\d+)/2', content)
                if match:
                    particle, n_val, l_char, j_half = match.group(1), int(match.group(2)), match.group(3).lower(), float(match.group(4))/2.0
                    label = f"{particle}.{n_val}{l_char}.{int(j_half*2)}/2"

                    matched = False
                    for ts, lj in target_ljs:
                        normed = label.replace('.', '').lower()
                        ts_lower = ts.lower()
                        # ★ 关键修复：精确匹配主量子数 n
                        # 例: '1s1/2' 应只匹配 N.1s.1/2，不匹配 N.2s.1/2
                        if ts_lower == normed or ts_lower == f"{n_val}{l_char}{int(j_half*2)}/2":
                            matched = True; break
                        # 兼容旧逻辑：纯 lj 匹配（无前缀数字）
                        if ts_lower.startswith(f"{l_char}") and lj is not None:
                            if lj[0] == l_char and abs(lj[1] - j_half) < 0.01:
                                matched = True; break

                    if matched:
                        # 简化：不再做复杂的收敛性检查，final即最终数据
                        # 仅做极端值过滤：最终loop的 g,f 不超过 1000
                        last_file = os.path.join(wav_dir, sorted(state_files, key=lambda x: _extract_it_loop(x))[-1])
                        if _check_extreme(last_file):
                            filtered.append(state)
                        else:
                            print(f"    ⚠️ 态 {state} ({label}) 最终loop极端值过大，已跳过")
            except Exception:
                continue

        print(f"🎯 目标态筛选: {self.target_states} → 匹配到 {len(filtered)} 个态")
        return filtered


class _RHF_Dataset(Dataset):
    """
    实际数据集：支持自适应序列长度 + 演化进度通道 + (Z,N) 宏观条件。

    输出格式（七元组）：
      X: (max_seq_len, 12, npt)  — 前11维物理场 + 第12维演化进度 progress∈[0,1]
      Y: (11, npt)               — 最终收敛态波函数（仅11物理通道，不含progress）
      kappa: (,)                  — κ量子数
      is_proton: (,)              — 粒子类型
      actual_seq_len: (,)         — 该样本实际有效序列长度
      z_num: (,)                  — 原子序数 Z
      n_num: (,)                  — 中子数 N
    """

    def __init__(self, data_dir, isotope, max_seq_len=10, min_seq_len=3,
                 traj_usage_ratio=0.8, target_states=None, allowed_states=None):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.traj_usage_ratio = traj_usage_ratio
        self.target_states = target_states
        self.allowed_states = allowed_states or set()

        # 获取该核素的 (Z, N)
        self.z_num, self.n_num = get_zn(isotope)

        wav_dir = os.path.join(data_dir, isotope, 'WAV')
        pot_dir = os.path.join(data_dir, isotope, 'POT')

        if not os.path.exists(wav_dir):
            return

        all_wav_files = sorted([f for f in os.listdir(wav_dir) if '.loop' in f])

        if not self.allowed_states:
            meta = _RHF_MetaDataset(data_dir, isotope, max_seq_len, min_seq_len,
                                     traj_usage_ratio, target_states)
            self.allowed_states = set(t[1] for t in meta.traj_meta)

        self.samples = []
        # ★ 完整的核素→文件前缀映射（覆盖全部37个核素）
        prefix_map = {
            '14O': 'O14_', '16O': 'O16_', '18O': 'O18_', '20O': 'O20_',
            '22O': 'O22_', '24O': 'O24_',
            '36Ca': 'Ca36_', '38Ca': 'Ca38_', '40Ca': 'Ca40_',
            '42Ca': 'Ca42_', '44Ca': 'Ca44_', '46Ca': 'Ca46_', '48Ca': 'Ca48_',
            '50Ca': 'Ca50_', '52Ca': 'Ca52_',
            '56Ni': 'Ni56_', '58Ni': 'Ni58_', '60Ni': 'Ni60_', '62Ni': 'Ni62_',
            '64Ni': 'Ni64_', '68Ni': 'Ni68_', '72Ni': 'Ni72_', '78Ni': 'Ni78_',
            '100Sn': 'Sn100_', '112Sn': 'Sn112_', '116Sn': 'Sn116_',
            '120Sn': 'Sn120_', '124Sn': 'Sn124_', '132Sn': 'Sn132_',
            '204Pb': 'Pb204_', '206Pb': 'Pb206_', '208Pb': 'Pb208_', '210Pb': 'Pb210_',
            '86Kr': 'Kr86_', '88Sr': 'Sr88_', '90Zr': 'Zr90_', '92Mo': 'Mo92_',
        }
        iso_prefix = prefix_map.get(isotope, isotope.split('/')[-1])
        matched_states = [s for s in self.allowed_states if s.startswith(iso_prefix)]

        for state in matched_states:
            state_files = sorted([f for f in all_wav_files if f.startswith(state)],
                                 key=lambda x: _extract_it_loop(x))

            # ★ 从 state 文件名或内容中提取主量子数 n
            n_principal = 1  # 默认值
            first_wav = os.path.join(wav_dir, state_files[0]) if state_files else None
            if first_wav and os.path.exists(first_wav):
                try:
                    with open(first_wav, 'r') as f:
                        content = f.read()
                    m = re.search(r'State:\s*[NP]\.(\d+)([a-z])\.(\d+)/2', content)
                    if m:
                        n_principal = int(m.group(1))
                except Exception:
                    pass

            # 加载完整轨迹
            trajectory_data = []
            kappa = -1.0
            is_proton = 0.0

            for wav_file in state_files:
                pot_file = wav_file.replace('.it', '_POT.it')
                res = _parse_single_step(os.path.join(wav_dir, wav_file),
                                         os.path.join(pot_dir, pot_file),
                                         data_dir)
                if res is not None:
                    step_tensor, parsed_kappa, parsed_proton = res
                    trajectory_data.append(step_tensor)
                    kappa = parsed_kappa
                    is_proton = parsed_proton

            L_total = len(trajectory_data)
            if L_total < self.min_seq_len + 1:
                continue

            # 堆叠成张量 (L_total, 11, npt)
            traj_tensor = torch.stack(trajectory_data)
            npt = traj_tensor.shape[-1]

            # ════════════════════════════════════════════════
            # 关键改动：Y_target 为最终收敛态波函数
            # 取该 state 排序后最后一个 loop（用户确认: final就是最终数据）
            # ════════════════════════════════════════════════
            Y_final = traj_tensor[-1]  # (11, npt) — 最终收敛态

            # 自适应序列长度
            eff_len = max(self.min_seq_len,
                          min(self.max_seq_len,
                              int(L_total * self.traj_usage_ratio)))

            # 全轨迹跨度采样
            n_windows = L_total - eff_len + 1
            if n_windows <= 0:
                starts = [0]
            elif n_windows <= eff_len:
                starts = list(range(n_windows))
            else:
                stride = max(1, n_windows // eff_len)
                starts = list(range(0, n_windows, stride))
                if starts[-1] != n_windows - 1:
                    starts.append(n_windows - 1)

            for start in starts:
                end = min(start + eff_len, L_total)
                actual_len = end - start

                # 提取子序列 (actual_len, 11, npt)
                X_11ch = traj_tensor[start:end]

                # Y目标：最终收敛态（所有窗口共享同一个 target）
                Y_11ch = Y_final

                # 第12通道：演化进度 progress ∈ [0, 1]
                indices = torch.arange(actual_len, dtype=torch.float32) + start
                progress_values = indices / max(L_total - 1, 1)
                progress_channel = progress_values.view(-1, 1, 1).expand(-1, 1, npt)

                # 拼接12维：(actual_len, 12, npt)
                X_12ch = torch.cat([X_11ch, progress_channel], dim=1)

                # Padding 到固定的 max_seq_len
                if actual_len < self.max_seq_len:
                    pad_len = self.max_seq_len - actual_len
                    last_frame = X_12ch[-1:].clone()
                    last_frame[:, 11, :] = 1.5  # progress>1.0 标记无效padding
                    pad_frames = last_frame.expand(pad_len, -1, -1).clone()

                    for p in range(pad_len):
                        pad_frames[p, 11, :] = 1.0 + (p + 1) * 0.1

                    X_padded = torch.cat([X_12ch, pad_frames], dim=0)
                else:
                    X_padded = X_12ch

                # 存储八元组（★ 新增 n_principal 主量子数）
                self.samples.append((
                    X_padded,           # (max_seq_len, 12, npt)
                    Y_11ch,             # (11, npt) — 最终收敛态（不含progress）
                    kappa,
                    is_proton,
                    actual_len,
                    self.z_num,         # 原子序数 Z
                    self.n_num,         # 中子数 N
                    float(n_principal),  # ★ 主量子数 n（区分 1s vs 2s vs 3s ...）
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==================== 向后兼容：保留原始 RHF_Dataset 接口 ====================
class RHF_Dataset(Dataset):
    """
    原始接口的向后兼容封装。
    """

    def __init__(self, data_dir, isotope, seq_len=5, mode='train', min_traj_len=None,
                 target_states=None):
        self._inner = _RHF_Dataset(
            data_dir=data_dir, isotope=isotope,
            max_seq_len=seq_len,
            min_seq_len=max(3, seq_len // 2),
            traj_usage_ratio=0.85,
            target_states=target_states
        )
        self.samples = self._inner.samples

    def __len__(self): return len(self._inner)
    def __getitem__(self, idx): return self._inner[idx]


# ==================== 工具函数 ====================

def _extract_it_loop(filename):
    match = re.search(r'\.it(\d+)\.loop(\d+)', filename)
    return int(match.group(1)) * 1000 + int(match.group(2)) if match else 0


def _fix_fortran_float(s):
    """修复 Fortran 科学计数法: '0.4215+213' → '0.4215E+213'"""
    if isinstance(s, bytes):
        s = s.decode('ascii', errors='ignore')
    s = s.strip()
    m = re.match(r'^([+-]?\d*\.?\d*)\+?(-?\d+)$', s)
    if m:
        base, exp = m.group(1), m.group(2)
        return float(f"{base}E{exp}")
    return float(s)


def _check_extreme(filepath):
    """
    极端值检查：对波函数先归一化再判断，而非直接检查原始值。

    ★ 关键改进：RHF原始数据量级差异大（不同核素/轨道可差10³倍），
              直接用绝对阈值会误杀有效数据。归一化后统一到 O(1) 量级再检查。
    """
    try:
        d = np.loadtxt(filepath, comments='#', converters={1: _fix_fortran_float, 2: _fix_fortran_float})
        r = d[:, 0]
        g, f = d[:, 1], d[:, 2]

        # ★ 先归一化（与 _parse_single_step 中完全一致的逻辑）
        norm_integral = np.trapz(g**2 + f**2, x=r)
        if norm_integral > 1e-12:
            nf = 1.0 / np.sqrt(norm_integral)
            g_norm = g * nf
            f_norm = f * nf
        else:
            return False  # 归一化积分为零 → 无效数据

        # 检查归一化后的值是否合理（O(1)量级）
        max_g = np.max(np.abs(g_norm))
        max_f = np.max(np.abs(f_norm))

        # 合理的束缚态：归一化后 |g| < 50, |f| < 50
        # （正常范围是 |g|~1~30, |f|~0.01~5；留余量防数值噪声）
        return max_g < 50.0 and max_f < 50.0
    except Exception:
        return False


def _check_converged(filepath):
    """
    保留旧接口兼容性，内部调用 _check_extreme。
    """
    return _check_extreme(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# PKA1 文件缓存：避免重复读取
# ═══════════════════════════════════════════════════════════════════════════
_pka1_cache = {}  # {(isotope, particle_type): (r_grid, state_names, G_data, F_data)}


def _load_pka1_data(data_dir, isotope, particle_type):
    """
    从 PKA1 文件加载最终收敛的波函数 G/F 分量。

    PKA1 文件格式：
      - 行0: 占据概率（43列：1列标识 + 42列态）
      - 行1: 列标题 r, N.1s.1/2, N.2s.1/2, ...
      - 行2+: 数据行（201个径向点，dr=0.1fm）

    参数:
      data_dir: 数据根目录
      isotope: 核素名（如 '16O'）
      particle_type: 'N'（中子）或 'P'（质子）

    返回:
      (r_grid, state_names, G_data, F_data) 或 None
      - r_grid: (201,) 径向网格
      - state_names: 状态名列表（如 ['N.1s.1/2', 'N.2s.1/2', ...]）
      - G_data: (n_states, 201) 大分量
      - F_data: (n_states, 201) 小分量
    """
    cache_key = (isotope, particle_type)
    if cache_key in _pka1_cache:
        return _pka1_cache[cache_key]

    # 构建 PKA1 文件路径
    prefix_map = {
        '14O': 'O14', '16O': 'O16', '18O': 'O18', '20O': 'O20',
        '22O': 'O22', '24O': 'O24',
        '36Ca': 'Ca36', '38Ca': 'Ca38', '40Ca': 'Ca40',
        '42Ca': 'Ca42', '44Ca': 'Ca44', '46Ca': 'Ca46', '48Ca': 'Ca48',
        '50Ca': 'Ca50', '52Ca': 'Ca52',
        '56Ni': 'Ni56', '58Ni': 'Ni58', '60Ni': 'Ni60', '62Ni': 'Ni62',
        '64Ni': 'Ni64', '68Ni': 'Ni68', '72Ni': 'Ni72', '78Ni': 'Ni78',
        '100Sn': 'Sn100', '112Sn': 'Sn112', '116Sn': 'Sn116',
        '120Sn': 'Sn120', '124Sn': 'Sn124', '132Sn': 'Sn132',
        '204Pb': 'Pb204', '206Pb': 'Pb206', '208Pb': 'Pb208', '210Pb': 'Pb210',
        '86Kr': 'Kr86', '88Sr': 'Sr88', '90Zr': 'Zr90', '92Mo': 'Mo92',
    }
    prefix = prefix_map.get(isotope, isotope)
    pka1_g_path = os.path.join(data_dir, isotope, 'WAV', f'{prefix}.G-{particle_type}.PKA1')
    pka1_f_path = os.path.join(data_dir, isotope, 'WAV', f'{prefix}.F-{particle_type}.PKA1')

    if not os.path.exists(pka1_g_path) or not os.path.exists(pka1_f_path):
        return None

    try:
        # 读取 G 文件（大分量）
        with open(pka1_g_path, 'r') as f:
            lines_g = f.readlines()

        # 解析列标题获取状态名
        header = lines_g[1].strip().split()
        state_names = header[1:]  # 跳过 'r'

        # 读取数据（跳过前2行）
        data_g = np.loadtxt(pka1_g_path, comments='#', skiprows=2)
        data_f = np.loadtxt(pka1_f_path, comments='#', skiprows=2)

        r_grid = data_g[:, 0]
        G_data = data_g[:, 1:].T  # (n_states, n_points)
        F_data = data_f[:, 1:].T  # (n_states, n_points)

        result = (r_grid, state_names, G_data, F_data)
        _pka1_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  ⚠️ 读取 PKA1 文件失败: {e}")
        return None


def _get_state_index_from_loop(wav_path):
    """
    从 loop 文件名或内容中提取状态索引（用于 PKA1 文件列查找）。

    返回: (state_name, n_principal) 如 ('N.1s.1/2', 1)
    """
    try:
        with open(wav_path, 'r') as f:
            content = f.read()

        # 从 State: 行提取信息
        match = re.search(r'State:\s*([NP])\.(\d+)([a-z])\.(\d+)/2', content)
        if match:
            particle = match.group(1)
            n_val = int(match.group(2))
            l_char = match.group(3)
            j_half = int(match.group(4))
            state_name = f"{particle}.{n_val}{l_char}.{j_half}/2"
            return state_name, n_val
    except Exception:
        pass
    return None, 1


def _parse_single_step(wav_path, pot_path, data_dir=None):
    """
    核心解析函数：提取 11 个物理通道和 kappa 量子数。

    ★ 关键修复：波函数从 PKA1 文件读取（最终收敛态），
       而非 loop 文件（被截断的 RHF 迭代数据）。

    返回: (tensor_11x201, kappa_float, is_proton_float) 或 None

    11个通道: [g, f, vps, vms, vtt, XG, XF, YG, YF, E_array, vv_array]
    """
    # ═══════════════════════════════════════════════════════════════════════
    # 步骤1: 从 loop 文件读取元信息（能量、占据、态标识）
    # ═══════════════════════════════════════════════════════════════════════
    with open(wav_path, 'r') as f:
        lines = f.readlines()

    energy_line = [l for l in lines if 'Energy=' in l][0]
    energy = float(re.search(r'Energy=\s*([-\d\.E]+)', energy_line).group(1))

    occ_line = [l for l in lines if 'Occupation probability:' in l][0]
    vv = float(re.search(r'Occupation probability:\s+([-\d\.E]+)', occ_line).group(1))

    state_line = [l for l in lines if 'State:' in l][0]
    match = re.search(r'State:\s*([NP])\.\d+([a-z])\.(\d+)/2', state_line)
    kappa = -1.0
    is_proton = 0.0
    state_name_from_loop = None
    if match:
        particle = match.group(1)
        l_char = match.group(2)
        j_num = float(match.group(3))
        is_proton = 1.0 if particle == 'P' else 0.0
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
        l_val = l_map.get(l_char, 0)
        j_val = j_num / 2.0
        kappa = -(l_val + 1) if j_val > l_val else l_val
        state_name_from_loop = f"{particle}.{match.group(0).split('.')[1]}.{int(j_num)}/2"

    # ═══════════════════════════════════════════════════════════════════════
    # 步骤2: 从 PKA1 文件读取正确的波函数 G/F 分量
    # ═══════════════════════════════════════════════════════════════════════
    # 推断 data_dir 从 wav_path
    if data_dir is None:
        # wav_path = .../results/16O/WAV/O16_state001.it001.loop001
        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(wav_path)))

    # 从 loop 文件名推断核素
    wav_dir = os.path.dirname(wav_path)
    isotope = os.path.basename(os.path.dirname(wav_dir))

    particle_type = 'P' if is_proton else 'N'

    # 加载 PKA1 数据
    pka1_result = _load_pka1_data(data_dir, isotope, particle_type)
    if pka1_result is None:
        # 回退到 loop 文件（如果 PKA1 不存在）
        print(f"  ⚠️ PKA1 文件不存在，回退到 loop 文件: {isotope} {particle_type}")
        wav_data = np.loadtxt(wav_path, comments='#', converters={1: _fix_fortran_float, 2: _fix_fortran_float})
        g, f = wav_data[:, 1], wav_data[:, 2]
    else:
        r_grid_pka, state_names, G_data, F_data = pka1_result

        # 从 loop 文件内容获取状态名
        state_name_loop, n_principal = _get_state_index_from_loop(wav_path)

        # 在 PKA1 状态列表中查找匹配
        if state_name_loop and state_name_loop in state_names:
            state_idx = state_names.index(state_name_loop)
            g = G_data[state_idx].copy()
            f = F_data[state_idx].copy()
        else:
            # 尝试模糊匹配（忽略大小写）
            state_name_lower = state_name_loop.lower() if state_name_loop else None
            found = False
            for i, name in enumerate(state_names):
                if name.lower() == state_name_lower:
                    g = G_data[i].copy()
                    f = F_data[i].copy()
                    found = True
                    break
            if not found:
                print(f"  ⚠️ 状态 {state_name_loop} 不在 PKA1 中，回退到 loop 文件")
                wav_data = np.loadtxt(wav_path, comments='#', converters={1: _fix_fortran_float, 2: _fix_fortran_float})
                g, f = wav_data[:, 1], wav_data[:, 2]

    npt = len(g)
    r_grid = np.arange(npt) * 0.10
    r_grid[0] = 0.0010

    # ═══════════════════════════════════════════════════════════════════════
    # 步骤3: 归一化波函数
    # ═══════════════════════════════════════════════════════════════════════
    norm_integral = np.trapz(g**2 + f**2, x=r_grid)

    if norm_integral > 1e-12:
        nf = 1.0 / np.sqrt(norm_integral)
        g, f = g * nf, f * nf
    else:
        return None

    # 极端值检查
    if np.max(np.abs(g)) > 50.0 or np.max(np.abs(f)) > 50.0:
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # 步骤4: 相位约定 - 保持 PKA1 原始相位（不再强制翻转）
    # ═══════════════════════════════════════════════════════════════════════
    # ★ 重要：PKA1 文件中的波函数已经是物理正确的相位
    # 对于 κ<0 的态（如 1s1/2），g>0, f<0 是正确的物理行为
    # 不再强制翻转相位，让网络学习真实的物理相位关系

    # ═══════════════════════════════════════════════════════════════════════
    # 步骤5: 从 POT 文件读取势场
    # ═══════════════════════════════════════════════════════════════════════
    pot_data = np.loadtxt(pot_path, comments='#')
    vps, vms, vtt = pot_data[:, 1], pot_data[:, 2], pot_data[:, 3]
    XG, XF, YG, YF = pot_data[:, 4], pot_data[:, 5], pot_data[:, 6], pot_data[:, 7]
    E_array, vv_array = np.full_like(g, energy), np.full_like(g, vv)

    tensor_np = np.stack([g, f, vps, vms, vtt, XG, XF, YG, YF, E_array, vv_array], axis=0)
    return torch.tensor(tensor_np, dtype=torch.float32), kappa, is_proton


if __name__ == "__main__":
    print("=== 数据管道测试（Y=最终收敛态 + ZN元数据 版）===")

    test_dir = '/home/ubuntu/rhf/results'
    # ★ 完整的42个核子态（与 Train.py 保持一致）
    targets = [
        '1s1/2', '2s1/2', '3s1/2', '4s1/2', '5s1/2', '6s1/2',
        '1p3/2', '2p3/2', '3p3/2', '4p3/2', '5p3/2', '6p3/2',
        '1d5/2', '2d5/2', '3d5/2', '4d5/2', '5d5/2',
        '1f7/2', '2f7/2', '3f7/2', '4f7/2', '5f7/2',
        '1p1/2', '2p1/2', '3p1/2', '4p1/2', '5p1/2', '6p1/2',
        '1d3/2', '2d3/2', '3d3/2', '4d3/2', '5d3/2',
        '1f5/2', '2f5/2', '3f5/2', '4f5/2', '5f5/2',
        '1g7/2', '2g7/2', '3g7/2', '4g7/2',
    ]
    isos = ['16O', '40Ca']

    # 测试1：新接口 build_datasets
    print("\n--- 测试 build_datasets (train/val/test 划分) ---")
    for m in ['train', 'val', 'test']:
        ds = build_datasets(test_dir, isos, max_seq_len=10, min_seq_len=3,
                           traj_usage_ratio=0.8, mode=m, target_states=targets)
        if ds is not None:
            print(f"  [{m}]: {len(ds)} samples")
            # 检查一个样本的shape
            sample = ds[0]
            print(f"       X shape: {sample[0].shape}, Y shape: {sample[1].shape}")
            print(f"       kappa={sample[2]}, is_proton={sample[3]}, actual_len={sample[4]}")
            print(f"       Z={sample[5]}, N={sample[6]}")
        else:
            print(f"  [{m}]: None")


# ═══════════════════════════════════════════════════════════════
#   ★ 新增：按核素分组的Batch采样器
#   确保同一batch内的样本来自同一核素（自注意力所需）
# ═══════════════════════════════════════════════════════════════

import torch.utils.data as torch_data


class IsotopeGroupedBatchSampler(torch_data.Sampler):
    """
    按核素分组的批采样器。

    核心功能：保证同一个batch内的所有样本来自同一个核素，
    这是 OrbitalSelfAttention 的输入前提——同核素轨道间才能互相attend。

    工作流程：
      1. 预扫描数据集，按 isotope 对样本索引分组
      2. 每次生成batch时，随机选择一个核素，从中采样batch_size个样本
      3. 如果某个核素样本不足batch_size，则取该核素全部样本

    参数：
      dataset: _RHF_Dataset 或 ConcatDataset 实例
      batch_size: 每个batch的样本数
      shuffle: 是否打乱核素顺序和组内样本
      isotope_key_fn: 从数据集样本中提取核素标识的函数
    """
    def __init__(self, dataset, batch_size, shuffle=True, isotope_key_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 构建核素→索引映射
        self.isotope_indices = {}
        self._build_groups(isotope_key_fn)

    def _build_groups(self, isotope_key_fn):
        """预扫描数据集，按核素分组"""
        if isotope_key_fn is not None:
            # 自定义分组函数
            for idx in range(len(self.dataset)):
                key = isotope_key_fn(self.dataset, idx)
                if key not in self.isotope_indices:
                    self.isotope_indices[key] = []
                self.isotope_indices[key].append(idx)
        else:
            # 默认：对ConcatDataset按子数据集分组（每个子集=一个核素）
            from torch.utils.data import ConcatDataset
            if isinstance(self.dataset, ConcatDataset):
                offset = 0
                for i, sub_ds in enumerate(self.dataset.datasets):
                    sub_len = len(sub_ds)
                    # 用子数据集索引作为分组key
                    key = i
                    # 尝试从子数据集获取核素名
                    if hasattr(sub_ds, 'z_num') and hasattr(sub_ds, 'n_num'):
                        key = f"Z{sub_ds.z_num}_N{sub_ds.n_num}"
                    self.isotope_indices[key] = list(range(offset, offset + sub_len))
                    offset += sub_len
            else:
                # 单一数据集：全部归为一组
                self.isotope_indices[0] = list(range(len(self.dataset)))

        # 过滤空组
        self.isotope_indices = {k: v for k, v in self.isotope_indices.items() if len(v) > 0}

    def __iter__(self):
        # 打乱核素顺序
        keys = list(self.isotope_indices.keys())
        if self.shuffle:
            np.random.shuffle(keys)

        # 对每个核素，打乱其内部样本顺序
        indices_per_isotope = {}
        for k in keys:
            idx_list = self.isotope_indices[k].copy()
            if self.shuffle:
                np.random.shuffle(idx_list)
            indices_per_isotope[k] = idx_list

        # 生成batch：从每个核素依次取batch_size个样本
        # 当某个核素样本耗尽时跳到下一个核素
        cursor = {k: 0 for k in keys}
        active_keys = keys.copy()

        while active_keys:
            # 随机选一个还有剩余样本的核素
            if self.shuffle:
                chosen_key = active_keys[np.random.randint(len(active_keys))]
            else:
                chosen_key = active_keys[0]

            start = cursor[chosen_key]
            end = min(start + self.batch_size, len(indices_per_isotope[chosen_key]))
            batch_indices = indices_per_isotope[chosen_key][start:end]
            cursor[chosen_key] = end

            if len(batch_indices) > 0:
                yield batch_indices

            # 如果该核素样本用完，移出活跃列表
            if cursor[chosen_key] >= len(indices_per_isotope[chosen_key]):
                active_keys.remove(chosen_key)

    def __len__(self):
        total_batches = 0
        for k, indices in self.isotope_indices.items():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


def scan_available_isotopes(data_dir):
    """
    扫描数据目录，返回所有可用核素列表。
    自动发现 ISOTOPE_ZN 中未列出的核素。
    """
    available = []
    if not os.path.exists(data_dir):
        return available

    for entry in sorted(os.listdir(data_dir)):
        iso_dir = os.path.join(data_dir, entry, 'WAV')
        if os.path.isdir(iso_dir) and len(os.listdir(iso_dir)) > 0:
            if entry in ISOTOPE_ZN:
                available.append(entry)
            else:
                # 尝试解析未知核素
                zn = get_zn(entry)
                if zn != (0, 0):
                    available.append(entry)
                    print(f"  ℹ️ 发现未注册核素 {entry} (Z={zn[0]}, N={zn[1]})，已自动加入")

    return available
