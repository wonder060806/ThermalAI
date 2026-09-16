"""
gen_cases_all.py — 统一生成四种冷却模式 3D-ICE case

模式: solid(固体), air(风冷), liquid(液冷冷板), microchannel(微通道)
每种 30 case, 共用相同芯片布局, BC不同。

3D-ICE 单位: 长度(μm), k(W/(μmK) = W/(mK)/1e6), HTC(W/(μm²K) = W/(m²K)/1e12)
"""
import os, json, numpy as np

# ═══════════════ 几何参数 ═══════════════
CHIP_UM = 1000
CELLS = 20
CELL_UM = CHIP_UM // CELLS
SRC_UM, BLK_UM, TOTAL_UM = 5, 495, 500

# ═══════════════ 材料 ═══════════════
K_TOY = 1.0e-7        # 0.1 W/(mK)
K_SILICON = 1.30e-4   # 130 W/(mK)
HC_TOY = 1.628e-12    # 体积热容 (不重要,稳态)

# ═══════════════ 功率 ═══════════════
NUM_PER_MODE = 30
P_TOTAL_MIN_MW = 1.0
P_TOTAL_MAX_MW = 8.0

# Per-mode power override (microchannel needs higher power to create gradients on k=130 Si)
MODE_POWER = {
    "microchannel": (20.0, 80.0),  # 10x higher — k=130 needs more power for meaningful dT
}
BLOCK_CELLS_MIN = 2
BLOCK_CELLS_MAX = 8
N_BLOCKS_MIN = 4
N_BLOCKS_MAX = 10

# ═══════════════ 路径 ═══════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
BIN_DIR = os.path.join(ROOT, "bin")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BIN_DIR, exist_ok=True)

# ═══════════════ 冷却模式定义 ═══════════════

MATERIAL_TOY = f"""material TOY :
   thermal conductivity     {K_TOY} ;
   volumetric heat capacity {HC_TOY} ;"""

MATERIAL_SILICON = f"""material SILICON :
   thermal conductivity     {K_SILICON} ;
   volumetric heat capacity {HC_TOY} ;"""

STK_BASE = """{material}

{top_bc}

bottom heat sink :
   heat transfer coefficient {htc_btm} ;
   temperature               {t_amb} ;

dimensions :
   chip length {chip}, width {chip} ;
   cell length  {cell}, width  {cell} ;
   non-uniform false;

die TOP_IC :
   source  {src} {mat_name} ;
   layer   {blk} {mat_name} ;

stack:
   die     MY_DIE     TOP_IC    floorplan "../data/{flp_name}" ;

solver:
   steady ;
   initial temperature {t_init} ;
   numofcores 1 ;

output:
   Tmap  ( MY_DIE, "../data/{temp_name}", final ) ;
"""

CONFIG = {
    "liquid": {
        "mat": MATERIAL_TOY, "mat_name": "TOY",
        "top_bc": """top heat sink :
   heat transfer coefficient {htc_top} ;
   temperature               {t_amb} ;""",
        "htc_top": 5.0e-9,  # 5000 W/(m2K)
        "htc_btm": 5.0e-10, "t_amb": 293.0, "t_init": 293.0,
        "top_is_heatsink": False,
    },
    "air": {
        "mat": MATERIAL_TOY, "mat_name": "TOY",
        "top_bc": """top heat sink :
   heat transfer coefficient {htc_top} ;
   temperature               {t_amb} ;""",
        "htc_top": 5.0e-11,  # 50 W/(m2K)
        "htc_btm": 5.0e-10, "t_amb": 313.0, "t_init": 313.0,
        "top_is_heatsink": False,
    },
    "microchannel": {
        "mat": MATERIAL_SILICON + "\n\n" + MATERIAL_TOY, "mat_name": "SILICON",
        "top_bc": """top heat sink :
   heat transfer coefficient {htc_top} ;
   temperature               {t_amb} ;""",
        "htc_top": 5.0e-9,  # 5000 W/(m2K) equivalent
        "htc_btm": 5.0e-10, "t_amb": 293.0, "t_init": 293.0,
        "top_is_heatsink": False,
    },
}

# solid 跳过 — 已有 gen_cases_v3.py 生成的 200 case


def generate_blocks(rng, total_power_mw):
    n = rng.randint(N_BLOCKS_MIN, N_BLOCKS_MAX + 1)
    blocks = []
    for i in range(n):
        wc = rng.randint(BLOCK_CELLS_MIN, BLOCK_CELLS_MAX + 1)
        hc = rng.randint(BLOCK_CELLS_MIN, BLOCK_CELLS_MAX + 1)
        w, h = wc * CELL_UM, hc * CELL_UM
        max_x = CHIP_UM - w; max_y = CHIP_UM - h
        x = rng.randint(0, max(1, max_x // CELL_UM)) * CELL_UM if max_x >= CELL_UM else 0
        y = rng.randint(0, max(1, max_y // CELL_UM)) * CELL_UM if max_y >= CELL_UM else 0
        blocks.append({'x': x, 'y': y, 'w': w, 'h': h, 'area_cells': wc * hc})
    total_area = sum(b['area_cells'] for b in blocks)
    for b in blocks:
        b['power_mw'] = total_power_mw * b['area_cells'] / total_area
        b['power_mw'] *= rng.uniform(0.7, 1.3)
        b['power_w'] = b['power_mw'] / 1000.0
    actual = sum(b['power_mw'] for b in blocks)
    for b in blocks:
        b['power_mw'] *= total_power_mw / actual
        b['power_w'] = b['power_mw'] / 1000.0
    return blocks


def compute_dimless_power(blocks):
    cu = CHIP_UM / 20  # 20x20 output grid
    pm = np.zeros((20, 20))
    for b in blocks:
        i0 = max(0, int(b['x']/cu)); i1 = min(20, int(np.ceil((b['x']+b['w'])/cu)))
        j0 = max(0, int(b['y']/cu)); j1 = min(20, int(np.ceil((b['y']+b['h'])/cu)))
        n = (i1-i0)*(j1-j0)
        if n > 0: pm[i0:i1, j0:j1] += b['power_mw'] / n
    return pm / 0.00625  # dimensionless


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="air,liquid,microchannel",
                    help="comma-separated: air,liquid,microchannel")
    ap.add_argument("--num", type=int, default=NUM_PER_MODE)
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",")]

    rng = np.random.RandomState(42)

    for mode in modes:
        cfg = CONFIG[mode]
        p_min, p_max = MODE_POWER.get(mode, (P_TOTAL_MIN_MW, P_TOTAL_MAX_MW))

        # Generate mode-specific block layouts and powers
        mode_rng = np.random.RandomState(42 + list(CONFIG.keys()).index(mode))
        all_blocks = []
        for i in range(args.num):
            total_pw = round(rng.uniform(p_min, p_max), 2)
            for attempt in range(50):
                blks = generate_blocks(rng, total_pw)
                break
            all_blocks.append(blks)

        mode_cases = []
        for cid in range(args.num):
            blks = all_blocks[cid]
            flp_name = f"case_{mode}_{cid}.flp"
            temp_name = f"case_{mode}_{cid}_temp.txt"
            stk_name = f"case_{mode}_{cid}.stk"

            # 写 .flp
            lines = []
            for bi, b in enumerate(blks):
                lines.append(f"Block{bi} :")
                lines.append(f"  position   {b['x']}, {b['y']} ;")
                lines.append(f"  dimension  {b['w']}, {b['h']} ;")
                lines.append(f"  power values {b['power_w']:.6f} ;")
                lines.append("")
            with open(os.path.join(DATA_DIR, flp_name), 'w') as f:
                f.write('\n'.join(lines))

            # 写 .stk
            top_bc = cfg["top_bc"].format(
                htc_top=cfg.get("htc_top", 0), t_amb=cfg["t_amb"], chip=CHIP_UM, cell=CELL_UM
            ) if cfg.get("htc_top") is not None and not cfg.get("top_is_heatsink") else cfg["top_bc"].format(
                chip=CHIP_UM, t_amb=cfg["t_amb"]
            )

            stk = STK_BASE.format(
                material=cfg["mat"],
                top_bc=top_bc,
                htc_btm=cfg["htc_btm"], t_amb=cfg["t_amb"],
                chip=CHIP_UM, cell=CELL_UM,
                src=SRC_UM, blk=BLK_UM,
                mat_name=cfg["mat_name"],
                flp_name=flp_name, temp_name=temp_name,
                t_init=cfg["t_init"],
            )
            with open(os.path.join(BIN_DIR, stk_name), 'w') as f:
                f.write(stk)

            dl = compute_dimless_power(blks)
            mask = dl > 0.01
            mode_cases.append({
                'id': cid, 'mode': mode,
                'chip_length': CHIP_UM, 'num_cells': CELLS,
                'total_thickness': TOTAL_UM, 'num_blocks': len(blks),
                'total_power_mw': sum(b['power_mw'] for b in blks),
                'dimless_power_min': float(dl[mask].min()) if mask.sum() > 0 else 0,
                'dimless_power_max': float(dl[mask].max()) if mask.sum() > 0 else 0,
                'blocks': [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'],
                            'power_mw': round(b['power_mw'], 4)} for b in blks],
            })

        with open(os.path.join(DATA_DIR, f"cases_{mode}_params.json"), 'w') as f:
            json.dump(mode_cases, f, indent=2)
        print(f"[{mode}] {args.num} cases generated → {DATA_DIR}/cases_{mode}_params.json")

    print(f"\nDone. Run in WSL:")
    for mode in modes:
        print(f"  cd {BIN_DIR} && for f in case_{mode}_*.stk; do ./3D-ICE-Emulator $f; done")
