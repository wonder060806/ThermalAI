"""
gen_cases_v3.py — 生成匹配 DeepOHeat 参数的 3D-ICE case

物理参数（与 DeepOHeat 预训练一致）:
  k = 0.1 W/(mK) = 1e-7 W/(umK)     (toy material)
  HTC_bottom = 500 W/(m2K) = 5e-10   (仅底面散热)
  T_amb = 298.15 K
  Chip = 1000um x 1000um x 500um
  Grid = 20x20 cells = 50um/cell

功率范围: 总功率 1-8 mW (使无量纲功率在 DeepOHeat 训练范围内)
  v_dimless = P_cell_mW / 0.00625
  1 unit = 0.00625 mW/cell → 400 cells → 2.5 mW total for uniform v=1
"""
import os, json, random, numpy as np

# ─── 物理常数 ───
K_TOY   = 1.0e-7   # W/(umK) = 0.1 W/(mK)
HTC_BOT = 5.0e-10  # W/(um2K) = 500 W/(m2K)
T_AMB   = 298.15   # K
CHIP_UM = 1000     # um
CELLS   = 20       # 20x20 grid
CELL_UM = CHIP_UM // CELLS  # 50 um
SRC_UM  = 5        # source layer thickness (um)
BLK_UM  = 495      # bulk layer thickness (um)
TOTAL_UM = SRC_UM + BLK_UM  # 500 um

# ─── 功率范围 ───
P_TOTAL_MIN = 0.5   # mW
P_TOTAL_MAX = 5.0   # mW
N_BLOCKS_MIN = 4
N_BLOCKS_MAX = 10
BLOCK_CELLS_MIN = 2   # 最小 block 边长 (cell 数), 2*50=100um
BLOCK_CELLS_MAX = 8   # 最大 block 边长, 8*50=400um

# ─── 路径 ───
ROOT    = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
BIN_DIR  = os.path.join(ROOT, "bin")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BIN_DIR, exist_ok=True)

NUM_CASES = 30

# ─── STK 模板 ───
STK_TEMPLATE = """material TOY :
   thermal conductivity     {k} ;
   volumetric heat capacity 1.628e-12 ;

top heat sink :
   heat transfer coefficient 0.0 ;
   temperature               {tamb} ;

bottom heat sink :
   heat transfer coefficient {htc} ;
   temperature               {tamb} ;

dimensions :
   chip length {chip}, width {chip} ;
   cell length  {cell}, width  {cell} ;
   non-uniform false;

die TOP_IC :
   source  {src} TOY ;
   layer   {blk} TOY ;

stack:
   die     MY_DIE     TOP_IC    floorplan "../data/{flp_name}" ;

solver:
   steady ;
   initial temperature {tamb} ;
   numofcores 1 ;

output:
   Tmap  ( MY_DIE, "../data/{temp_name}", final ) ;
"""


def generate_blocks(rng, chip_um, total_power_mw):
    """生成不重叠的随机 block 布局, 总功率 = total_power_mw (mW).

    使用功率密度策略: 每个 block 的功率正比于其面积, 使无量纲功率值均匀.
    v_target = 目标无量纲功率 ~5-15 (在 DeepOHeat 可接受范围).
    """
    n = rng.randint(N_BLOCKS_MIN, N_BLOCKS_MAX + 1)
    blocks = []

    # 先生成所有 block 的尺寸和位置
    for i in range(n):
        w_cells = rng.randint(BLOCK_CELLS_MIN, BLOCK_CELLS_MAX + 1)
        h_cells = rng.randint(BLOCK_CELLS_MIN, BLOCK_CELLS_MAX + 1)
        w = w_cells * CELL_UM
        h = h_cells * CELL_UM

        max_x = chip_um - w
        max_y = chip_um - h
        x = rng.randint(0, max(1, max_x // CELL_UM)) * CELL_UM if max_x >= CELL_UM else 0
        y = rng.randint(0, max(1, max_y // CELL_UM)) * CELL_UM if max_y >= CELL_UM else 0

        area_cells = w_cells * h_cells
        blocks.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'area_cells': area_cells,
        })

    # 功率按面积比例分配 (每个 cell 无量纲功率 ~ v_target)
    total_area = sum(b['area_cells'] for b in blocks)
    for b in blocks:
        # 面积越大, 功率越大 (保持功率密度一致)
        b['power_mw'] = total_power_mw * b['area_cells'] / total_area
        # 加一些随机扰动 (±30%)
        b['power_mw'] *= rng.uniform(0.7, 1.3)
        b['power_w'] = b['power_mw'] / 1000.0

    # 重新归一化到 total_power_mw
    actual_total = sum(b['power_mw'] for b in blocks)
    for b in blocks:
        b['power_mw'] *= total_power_mw / actual_total
        b['power_w'] = b['power_mw'] / 1000.0

    return blocks


def blocks_overlap(b1, b2):
    """检查两个 block 是否重叠"""
    return not (
        b1['x'] + b1['w'] <= b2['x'] or
        b2['x'] + b2['w'] <= b1['x'] or
        b1['y'] + b1['h'] <= b2['y'] or
        b2['y'] + b2['h'] <= b1['y']
    )


def ensure_no_overlap(blocks):
    """调整重叠的 block 位置"""
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if blocks_overlap(blocks[i], blocks[j]):
                # 移动 j
                blocks[j]['x'] = (blocks[j]['x'] + blocks[j]['w'] + CELL_UM) % (CHIP_UM - blocks[j]['w'])
                blocks[j]['y'] = (blocks[j]['y'] + blocks[j]['h'] + CELL_UM) % (CHIP_UM - blocks[j]['h'])


def write_flp(blocks, path):
    """写 .flp floorplan 文件"""
    lines = []
    for i, blk in enumerate(blocks):
        lines.append(f"Block{i} :")
        lines.append(f"  position   {blk['x']}, {blk['y']} ;")
        lines.append(f"  dimension  {blk['w']}, {blk['h']} ;")
        lines.append(f"  power values {blk['power_w']:.6f} ;")
        lines.append("")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def compute_dimensionless_power(blocks, grid_size=CELLS):
    """计算 20x20 网格的无量纲功率分布"""
    cell_um = CHIP_UM / grid_size
    power_mw = np.zeros((grid_size, grid_size))

    for blk in blocks:
        i0 = max(0, int(blk['x'] / cell_um))
        i1 = min(grid_size, int(np.ceil((blk['x'] + blk['w']) / cell_um)))
        j0 = max(0, int(blk['y'] / cell_um))
        j1 = min(grid_size, int(np.ceil((blk['y'] + blk['h']) / cell_um)))
        n_cells = (i1 - i0) * (j1 - j0)
        if n_cells > 0:
            p_per_cell = blk['power_mw'] / n_cells
            power_mw[i0:i1, j0:j1] += p_per_cell

    # 无量纲化: v = P_cell_mW / 0.00625
    power_dimless = power_mw / 0.00625
    return power_mw, power_dimless


# ─── 主流程 ───
rng = np.random.RandomState(42)
cases = []

for cid in range(NUM_CASES):
    total_power_mw = round(rng.uniform(P_TOTAL_MIN, P_TOTAL_MAX), 2)

    # 尝试生成不重叠的 blocks
    for attempt in range(50):
        blocks = generate_blocks(rng, CHIP_UM, total_power_mw)
        ensure_no_overlap(blocks)
        # 验证无重叠
        ok = True
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                if blocks_overlap(blocks[i], blocks[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break

    # 计算无量纲功率统计
    power_mw, power_dimless = compute_dimensionless_power(blocks)
    dimless_active = power_dimless[power_dimless > 0.01]

    # 写文件
    flp_name = f"case_{cid}.flp"
    temp_name = f"case_{cid}_temp.txt"
    stk_name = f"case_{cid}.stk"

    write_flp(blocks, os.path.join(DATA_DIR, flp_name))

    stk_content = STK_TEMPLATE.format(
        k=K_TOY, htc=HTC_BOT, tamb=T_AMB,
        chip=CHIP_UM, cell=CELL_UM,
        src=SRC_UM, blk=BLK_UM,
        flp_name=flp_name, temp_name=temp_name,
    )
    with open(os.path.join(BIN_DIR, stk_name), 'w') as f:
        f.write(stk_content)

    case_info = {
        'id': cid,
        'chip_length': CHIP_UM,
        'num_cells': CELLS,
        'cell_length': CELL_UM,
        'total_thickness': TOTAL_UM,
        'source_thickness': SRC_UM,
        'num_blocks': len(blocks),
        'total_power_mw': total_power_mw,
        'dimless_power_min': float(dimless_active.min()) if len(dimless_active) > 0 else 0,
        'dimless_power_max': float(dimless_active.max()) if len(dimless_active) > 0 else 0,
        'dimless_power_mean': float(dimless_active.mean()) if len(dimless_active) > 0 else 0,
        'blocks': [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'],
                     'power_mw': round(b['power_mw'], 4)} for b in blocks],
    }
    cases.append(case_info)

    print(f"case_{cid:02d}: {len(blocks)} blocks, {total_power_mw} mW total, "
          f"dimless power [{case_info['dimless_power_min']:.2f}, {case_info['dimless_power_max']:.2f}]")

# 保存参数
with open(os.path.join(DATA_DIR, "cases_params.json"), 'w') as f:
    json.dump(cases, f, indent=2)

print(f"\n生成完成: {NUM_CASES} cases in {DATA_DIR}/ and {BIN_DIR}/")
print(f"准备在 WSL 中批量运行: cd /mnt/d/claudess/3d-ice/bin && for i in {{0..{NUM_CASES-1}}}; do ./3D-ICE-Emulator case_$i.stk; done")
