"""
run_20folds.py — 四模式 × 5折 = 20任务, 4卡自动调度

架构: MultiBranchDeepONet (4 branch + 1 trunk)
每任务: 一个模式的一折, 随机初始化, 纯监督训练
跑完: 每模式取最优折的 branch, 合并 → 一个模型
"""
import torch, os, sys, json, numpy as np, time, concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import subprocess

SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SCRIPT)  # ThermalAI-full root
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
PYTHON = sys.executable

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--modes", type=str, default="solid,air,liquid,microchannel",
                help="逗号分隔, 如 air,liquid")
ap.add_argument("--epochs", type=int, default=2000)
args_cli = ap.parse_args()

MODES = [m.strip() for m in args_cli.modes.split(",")]
EPOCHS = args_cli.epochs
GPUS = [0, 1, 2, 3]
LOG_DIR = os.path.join(SCRIPT, "log", "20folds")
os.makedirs(LOG_DIR, exist_ok=True)


def train_one_fold(mode, fold, gpu):
    """单进程跑 (mode, fold), 保存结果到 JSON"""
    t0 = time.time()
    tag = f"{mode}_f{fold}"
    log_file = os.path.join(LOG_DIR, f"{tag}.log")
    result_file = os.path.join(LOG_DIR, f"{tag}.json")

    cmd = [
        PYTHON, "-u", os.path.join(SCRIPT, "train_one_fold.py"),
        "--mode", mode, "--fold", str(fold), "--epochs", str(EPOCHS),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{tag}] GPU {gpu} start")
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=SCRIPT, text=True, bufsize=1, env=env)
        for line in proc.stdout:
            print(f"  [{tag}] {line.rstrip()}", flush=True)
            f.write(line)
        proc.wait()

    # 读结果
    delta = None
    if os.path.exists(result_file):
        with open(result_file) as f:
            d = json.load(f)
            delta = d.get("best_val_delta")
    elapsed = time.time() - t0
    print(f"  [{tag}] done: {delta:.2f}K ({elapsed:.0f}s)" if delta
          else f"  [{tag}] FAILED ({elapsed:.0f}s)")
    return mode, fold, delta


print("=" * 60)
print(f"20-fold training: {len(MODES)} modes x 5 folds, {len(GPUS)} GPUs")
print("=" * 60)

# 任务队列: 20 个 (mode, fold) 任务
tasks = [(m, f) for m in MODES for f in range(5)]
results = {}

with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
    pending = {}
    # 第一波: 每卡一个任务
    for g in GPUS:
        if tasks:
            mode, fold = tasks.pop(0)
            f = ex.submit(train_one_fold, mode, fold, g)
            pending[f] = (mode, fold, g)
            print(f"  Dispatch: {mode}_f{fold} -> GPU {g}")

    # 谁先完谁接下一个
    while pending:
        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            mode, fold, gpu_free = pending.pop(future)
            m, fo, delta = future.result()
            results[(m, fo)] = delta
            if tasks:
                nm, nf = tasks.pop(0)
                f = ex.submit(train_one_fold, nm, nf, gpu_free)
                pending[f] = (nm, nf, gpu_free)
                print(f"  Dispatch: {nm}_f{nf} -> GPU {gpu_free} (freed)")

# ═══ 汇总 ═══
print("\n" + "=" * 60)
print("Per-mode results:")
print("=" * 60)
best_per_mode = {}
for mode in MODES:
    folds = [results[(mode, f)] for f in range(5) if (mode, f) in results and results[(mode, f)] is not None]
    if folds:
        mean = np.mean(folds); std = np.std(folds)
        best_idx = np.argmin(folds)
        best_per_mode[mode] = {"mean": mean, "std": std, "best_fold": best_idx, "best_val": folds[best_idx]}
        print(f"  {mode:>15}: {mean:.2f} ± {std:.2f} K  (best fold {best_idx}: {folds[best_idx]:.2f}K)")
    else:
        print(f"  {mode:>15}: NO RESULTS")

# 合并: 加载已有 final 或新建, 只替换本次训练的模式
from multi_branch_model import MultiBranchDeepONet

ALL_MODES = ["solid", "air", "liquid", "microchannel"]
final_pth = os.path.join(LOG_DIR, "model_final.pth")
if os.path.exists(final_pth):
    final = MultiBranchDeepONet(ALL_MODES, device="cpu")
    sd = torch.load(final_pth, map_location="cpu", weights_only=True)
    # 处理 key 可能不匹配 (旧模型可能缺少某些 branch)
    final.load_state_dict(sd, strict=False)
    print(f"Loaded existing final model from {final_pth}")
else:
    final = MultiBranchDeepONet(ALL_MODES, device="cpu")
    # 从第一个本次训练的 mode 加载 trunk + 其他 branch
    any_pth = os.path.join(LOG_DIR, f"model_{MODES[0]}_f{best_per_mode[MODES[0]]['best_fold']}.pth")
    if os.path.exists(any_pth):
        final.load_state_dict(torch.load(any_pth, map_location="cpu", weights_only=True), strict=False)
        print(f"Initialized trunk from {any_pth}")

for mode in MODES:
    if mode in best_per_mode:
        bf = best_per_mode[mode]["best_fold"]
        pth = os.path.join(LOG_DIR, f"model_{mode}_f{bf}.pth")
        if os.path.exists(pth):
            sd = torch.load(pth, map_location="cpu", weights_only=True)
            for k in sd:
                if f"branches.{mode}" in k:
                    final.state_dict()[k].copy_(sd[k])
            print(f"  merged branch_{mode} from fold {bf} ({best_per_mode[mode]['best_val']:.2f}K)")

final_pth = os.path.join(LOG_DIR, "model_final.pth")
torch.save(final.state_dict(), final_pth)
print(f"\nFinal: {final_pth}")
