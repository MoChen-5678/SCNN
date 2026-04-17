import os

pot_dir = './results/16O/POT'
try:
    files = sorted([f for f in os.listdir(pot_dir) if f.endswith('.loop')])
    print(f"[STATUS] POT 目录检测完毕，共追踪到 {len(files)} 步自洽迭代历史。")
    if files:
        print("最新 5 步迭代文件：")
        for f in files[-5:]:
            print(f" -> {f}")
except FileNotFoundError:
    print(f"[ERROR] 路径 {pot_dir} 不存在。请检查前置的 Fortran 基准数据生成任务是否完成。")