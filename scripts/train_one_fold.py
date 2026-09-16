"""
train_one_fold.py — 训练单个 (mode, fold) 的 MultiBranchDeepONet

用法: py -3.12 train_one_fold.py --mode solid --fold 0 --epochs 2000
"""
import torch, os, sys, json, numpy as np, time, argparse, random

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_SCRIPT)  # ThermalAI-full root
os.chdir(_PROJECT)
sys.path.insert(0, _PROJECT)
from scipy.interpolate import griddata
from multi_branch_model import MultiBranchDeepONet

DATA_ROOT = os.path.join(_PROJECT, "data")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GRID_SIZE, P_PER_UNIT, T_REF, DT = 21, 0.00625, 293.15, 25.0
MODES = ["solid", "air", "liquid", "microchannel"]


def make_coords():
    from src import dataio_utils
    mesh = dataio_utils.fixed_mesh_grid_3d(starts=[0,0], ends=[1,1], num_intervals=[20,20])
    return torch.tensor(np.hstack([mesh, np.ones((len(mesh),1))*0.5]), dtype=torch.float32, device=DEVICE)


def load_case(mode, cid):
    cid = int(cid)
    if mode == "solid": tn, fn = f"case_{cid}_temp.txt", f"case_{cid}.flp"
    else: tn, fn = f"case_{mode}_{cid}_temp.txt", f"case_{mode}_{cid}.flp"
    with open(os.path.join(DATA_ROOT, tn)) as f:
        temps = []
        for line in f:
            if not line.startswith('%') and line.strip():
                temps.extend([float(x) for x in line.split()])
    temp_20 = np.array(temps).reshape(20, 20)
    src = np.linspace(0,1,20); dst = np.linspace(0,1,GRID_SIZE)
    sx,sy = np.meshgrid(src,src); dx,dy = np.meshgrid(dst,dst)
    temp_21 = griddata(np.column_stack([sx.ravel(),sy.ravel()]), temp_20.ravel(),
                       np.column_stack([dx.ravel(),dy.ravel()]), method='linear').reshape(GRID_SIZE,GRID_SIZE)
    with open(os.path.join(DATA_ROOT, fn)) as f:
        cur, blocks = {}, []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.endswith(':'):
                if cur and 'power' in cur: blocks.append(cur); cur = {}
            elif 'position' in line:
                p = line.replace(';','').replace(',',' ').split(); cur['x'],cur['y']=float(p[1]),float(p[2])
            elif 'dimension' in line:
                p = line.replace(';','').replace(',',' ').split(); cur['w'],cur['h']=float(p[1]),float(p[2])
            elif 'power values' in line:
                p = line.replace(';','').replace(',',' ').split(); cur['power']=float(p[2])
        if cur and 'power' in cur: blocks.append(cur)
    cu = 1000.0/GRID_SIZE; pm = np.zeros((GRID_SIZE,GRID_SIZE))
    for b in blocks:
        i0=max(0,int(b['x']/cu)); i1=min(GRID_SIZE,int(np.ceil((b['x']+b['w'])/cu)))
        j0=max(0,int(b['y']/cu)); j1=min(GRID_SIZE,int(np.ceil((b['y']+b['h'])/cu)))
        n=(i1-i0)*(j1-j0)
        if n>0: pm[i0:i1,j0:j1] += b['power']*1000.0/n
    return (pm/P_PER_UNIT, temp_21, (temp_21-T_REF)/DT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    mode = args.mode
    fold = args.fold

    # 加载数据
    if mode == "solid": pf = os.path.join(DATA_ROOT, "cases_params.json")
    else: pf = os.path.join(DATA_ROOT, f"cases_{mode}_params.json")
    with open(pf) as f: cases = json.load(f)
    keys = [(mode, int(c['id'])) for c in cases]

    # 5-fold 划分 (固定 seed)
    rng = random.Random(123)
    rng.shuffle(keys)
    n = len(keys)
    fs = n // 5
    folds = [keys[i*fs:(i+1)*fs] for i in range(5)]

    val_keys = folds[fold]
    train_keys = []
    for i in range(5):
        if i != fold: train_keys.extend(folds[i])

    print(f"Mode={mode} Fold={fold}: train={len(train_keys)} val={len(val_keys)}")

    # 加载 case 数据到内存
    case_data = {}
    for m, cid in train_keys + val_keys:
        if (m, cid) not in case_data:
            case_data[(m, cid)] = load_case(m, cid)

    coords = make_coords()
    model = MultiBranchDeepONet(MODES, device=DEVICE)
    mse = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_val, best_state = 999, None
    gradient_accum = args.grad_accum
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0, 0
        opt.zero_grad()
        rng.shuffle(train_keys)

        for m, cid in train_keys:
            p, tk, tu = case_data[(m, cid)]
            beta = torch.tensor(p.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(coords.shape[0], 1)
            target = torch.tensor(tu.flatten(), dtype=torch.float32, device=DEVICE)
            u = model(coords, beta, mode=m)
            loss = mse(u.flatten(), target) / gradient_accum
            loss.backward(); n_batches += 1
            epoch_loss += loss.item() * gradient_accum
            if n_batches % gradient_accum == 0:
                opt.step(); opt.zero_grad()
                n_batches = 0
        if n_batches > 0: opt.step(); opt.zero_grad()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_deltas = []
            with torch.no_grad():
                for m, cid in val_keys:
                    p, tk, tu = case_data[(m, cid)]
                    beta = torch.tensor(p.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(coords.shape[0], 1)
                    u = model(coords, beta, mode=m)
                    Tp = T_REF + DT * u.cpu().numpy().flatten()
                    tk_flat = T_REF + DT * tu.flatten()
                    mask = p.flatten() > 0.1
                    d = np.abs(Tp[mask]-tk_flat[mask]).mean() if mask.sum()>5 else np.abs(Tp-tk_flat).mean()
                    val_deltas.append(d)
            val_d = np.mean(val_deltas)
            model.train()
            if val_d < best_val:
                best_val = val_d
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch:4d}: loss={epoch_loss/len(train_keys):.4f} val={val_d:.2f}K")

    print(f"  Done: best_val={best_val:.2f}K ({time.time()-t0:.0f}s)")

    # 保存
    tag = f"{mode}_f{fold}"
    log_dir = os.path.join(_SCRIPT, "log", "20folds")
    os.makedirs(log_dir, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(log_dir, f"model_{tag}.pth"))
    with open(os.path.join(log_dir, f"{tag}.json"), 'w') as f:
        json.dump({"mode": mode, "fold": fold, "best_val_delta": round(best_val, 4),
                   "epochs": args.epochs, "n_train": len(train_keys), "n_val": len(val_keys)}, f, indent=2)
