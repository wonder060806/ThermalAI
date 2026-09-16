"""
heatsink_microchannel.py — 3D-ICE 微通道液冷散热器插件 (Python)

物理模型:
  - 并联矩形微通道, 层流强制对流换热
  - Nu = 4.36 (充分发展层流, 等热流边界)
  - h = Nu * k_coolant / Dh
  - 冷却液沿流向升温: T_coolant(x) = T_inlet + Q_accumulated(x) / (mdot * cp)
"""
import sys

# ── 从 args 解析参数 ──
# args 格式: "channel_width channel_height channel_pitch flow_rate t_coolant"
#   单位: μm, μm, μm, ml/min, K

def parse_args(args_str):
    tokens = args_str.split()
    params = {
        'channel_width':  float(tokens[0]),  # μm
        'channel_height': float(tokens[1]),  # μm
        'channel_pitch':  float(tokens[2]),  # μm
        'flow_rate':      float(tokens[3]),  # ml/min
        't_coolant_in':   float(tokens[4]),  # K
    }
    return params

# ── 全局状态 (heatsinkInit 设置) ──
params = {}
n_rows, n_cols = 0, 0
cell_w, cell_l = 0.0, 0.0
conductance_per_cell = 0.0
t_coolant_local = []  # 每列冷却液温度

# 水的物性 (300K)
RHO_WATER = 998.0       # kg/m³
CP_WATER = 4182.0       # J/(kg·K)
K_WATER = 0.598         # W/(m·K) at 293K
MU_WATER = 1.002e-3     # Pa·s
NU_LAMINAR = 4.36       # 等热流边界充分发展层流

def parallel(x, y):
    return x * y / (x + y) if (x + y) > 0 else 0


def heatsinkInit(nRows, nCols, cellWidth, cellLength,
                 initialTemperature, spreaderConductance, timeStep, args):
    """
    3D-ICE 调用。参数:
      nRows, nCols:           散热器覆盖的网格行列数
      cellWidth, cellLength:  每个cell尺寸 (μm)
      initialTemperature:     初始温度 (K)
      spreaderConductance:   散热器底板热导 (spreader k/L, W/(m²K))
      timeStep:               仿真时间步长 (s)
      args:                   用户参数字符串
    """
    global params, n_rows, n_cols, cell_w, cell_l
    global conductance_per_cell, t_coolant_local

    n_rows, n_cols = nRows, nCols
    cell_w, cell_l = cellWidth, cellLength

    params = parse_args(args)
    ch_w = params['channel_width'] * 1e-6    # μm → m
    ch_h = params['channel_height'] * 1e-6   # μm → m
    ch_p = params['channel_pitch'] * 1e-6    # μm → m

    # ── 水力直径 ──
    Dh = 2 * ch_w * ch_h / (ch_w + ch_h) if (ch_w + ch_h) > 0 else 1e-6

    # ── 每通道换热系数 ──
    h_channel = NU_LAMINAR * K_WATER / Dh  # W/(m²K)

    # ── 通道数 ──
    # 每个 cell 覆盖 chip_length/n_cols 宽度, pitch 决定通道密度
    cell_width_m = cellWidth * 1e-6  # μm → m
    n_channels_per_cell = max(1, cell_width_m / ch_p)

    # ── 每通道换热面积 ──
    cell_m = cellWidth * cellLength * 1e-12  # μm² → m²
    ch_perimeter = 2 * (ch_w + ch_h)  # m, 每通道润湿周长
    area_per_channel = ch_perimeter * cellWidth * 1e-6  # m²

    # ── 每cell等效换热系数 ──
    total_area = area_per_channel * n_channels_per_cell
    h_equiv = h_channel * total_area / cell_m if cell_m > 0 else h_channel

    # 串联热阻: 1/U = 1/h_equiv + 1/spreaderConductance
    U = parallel(h_equiv, spreaderConductance)

    # 每cell热导
    conductance_per_cell = U * cell_m  # W/K

    # ── 质量流量 ──
    flow_rate_m3s = params['flow_rate'] * 1e-6 / 60.0  # ml/min → m³/s
    mdot = flow_rate_m3s * RHO_WATER  # kg/s

    # ── 每列冷却液初始温度 ──
    t_coolant_local = [initialTemperature] * nCols

    print(f"[heatsink_microchannel] Init: {nRows}x{nCols}, "
          f"h={h_channel:.0f} W/m2K, U={U:.0f} W/m2K, "
          f"cond_per_cell={conductance_per_cell:.6f} W/K, mdot={mdot:.6f} kg/s",
          file=sys.stderr)


def heatsinkSimulateStep(spreaderTemperatures):
    """
    每步模拟: 接收spreader各cell温度, 返回各cell热流量 (W)。

    spreaderTemperatures: 长度 nRows*nCols 的列表, 按行排列。

    物理:
      - 每个cell的散热量 = conductance * (T_cell - T_coolant_col)
      - 冷却液每列升温 = 该列总散热量 / (mdot_per_col * cp)
      - 冷却液流向: x方向, 从col 0流向col nCols-1
    """
    global t_coolant_local

    # 质量流量分配到每行
    mdot_per_row = (RHO_WATER * params['flow_rate'] * 1e-6 / 60.0) / n_rows
    cp_per_row = mdot_per_row * CP_WATER  # W/K per row

    heat_flows = []
    total_heat = 0.0

    for row in range(n_rows):
        # 重置该行冷却液温度
        t_coolant_col = params['t_coolant_in']
        for col in range(nCols):
            idx = row * nCols + col
            T_cell = spreaderTemperatures[idx]

            # 散热量 (正值=散热, 负值=加热)
            q = conductance_per_cell * (T_cell - t_coolant_col)
            heat_flows.append(q)
            total_heat += q

            # 冷却液升温
            if cp_per_row > 0:
                t_coolant_col += q / cp_per_row

    return heat_flows
