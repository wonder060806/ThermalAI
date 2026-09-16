# ThermalAI 报告修订技术审计

## 结论边界

本项目是用神经算子学习“功率分布到温度场”的代理模型，并研究 PINN 预训练在小样本微调时是否有帮助。修订实验分成两条不能混淆的轨道：

1. 原尺度样本效率轨道：保持旧任务定义，用 10/20/50/100/200 个训练案例、公用隔离测试集和成对重复实验，只回答预训练是否提高样本效率。
2. 代表性工程尺度 3D 轨道：20 mm × 20 mm、150–500 W、三层稳态 3D-ICE、高 HTC 顶面等效边界，只回答该代理模型能否在较合理的面积、功率和体温度场上工作。

第二条不是 H100、B200 或 MI300X 的精确数字孪生，也没有显式求解流道内 CFD。任何报告文字不得写成“完整模拟了现代 AI 芯片封装”或“适用于全部冷却模式”。

## 全项目创新点与依据审查

| 项目改动 | 依据与判断 | 必须保留的限制 |
|---|---|---|
| 20 mm × 20 mm 域 | GH100 裸片面积公开为 814 mm²；20 mm 方形为 400 mm²，与现代大型算力裸片同数量级，比 1 mm 方形合理 | 不是任何商业芯片的真实长宽或 floorplan |
| 150–500 W | H100 SXM TDP 可到 700 W，MI300X 最大 TBP 750 W；本范围对应 37.5–125 W/cm²，覆盖 H100 约 86 W/cm² 的面积平均量级 | TDP/TBP 不等于全部热量均匀进入所建三层裸片 |
| 100000 W/(m²K) 顶部 HTC | IBM 300 mm 硅微通道冷却器报告约 104000 W/(m²K) 的表观 HTC，故数值数量级有实验依据 | 这是高性能微通道的等效边界，不代表普通冷板，更不代表风冷 |
| 三层 3D-ICE 真值 | 3D-ICE 使用有限体积热模型，论文报告其液冷 3D IC 验证平均误差低于 10%，适合早期设计探索 | 三层厚度 5/50/100 μm 是简化假设；必须做厚度、HTC 敏感性，不能称封装级认证模型 |
| 8 个异构矩形热区 | 用于生成可控且功率守恒的空间异质性，比单一均匀热源更能测试热点预测 | 不代表商业芯片专有模块布局；只可称合成异构热源 |
| DeepONet 结构 | 分支网络编码输入函数、主干网络编码查询坐标，符合原始 DeepONet 的算子学习定义 | 当前实现效果仍须由隔离测试集结果证明，结构名称本身不是性能证据 |
| PINN 损失权重扫描 | PINN 文献已知不同损失项存在梯度失衡和收敛速率不一致；因此不能凭一个权重断言物理约束“有害” | 权重扫描是预先声明的比较；同时报告梯度余弦、冲突比例及 Holm 校正后的统计结果 |
| 重复与统计 | 10/20 样本用 10 次，其余用 5 次；每次重新抽取嵌套训练子集，同时 PINN/随机初始化严格共用 data seed 和 init seed | 测试集在所有方法间固定并按布局组隔离；不能把同一训练子集的重复当成数据不确定性 |
| 微通道泄露审计 | 按几何布局分组划分，计算跨划分最近邻距离，并与常量场、总功率回归基线比较 | 极低误差只有通过独立生成的完整数据和这些门禁后才可报告为模型能力 |
| 相对误差 | 同时报 MAE、RMSE、最大误差、相对温升 MAE、峰值温升相对误差和热点位置误差 | 环境温度附近相对误差分母不稳定，因此以整体温升或峰值温升作分母并明确公式 |
| 推理速度 | 分开测模型加载、预热后单案例 GPU 推理以及外部 3D-ICE 进程；CUDA 计时前后同步 | 只能由同一工作站实测 JSON 计算加速比；本地冒烟不能支持“数千倍”结论 |

## 已发现并修正的实现问题

- 物理损失原先漏了前/后两个绝热边界，现六个外边界均有对应条件。
- 早期修订把旧物理项误写成“零源 PDE + 顶面绝热”，这并不等于官方 DeepOHeat。正式矩阵现改为两种有明确含义的输入：`surface_flux` 复现官方顶面 Neumann 功率，`volumetric_source` 匹配 3D-ICE 顶部 5 μm 发热层；`source_free` 只保留为可选负控制，不进入正式 110 任务。
- 物理/data Loss 的参考尺度原先取首个训练案例，导致权重受样本顺序影响；现训练前用全部训练案例的初始原始 Loss 均值校准并冻结。
- 小样本 3D-ICE 生成器原先把 5 μm source 声明在 die 底部，紧邻 bottom HTC，与 DeepOHeat 顶面功率不一致；现层序改为 495 μm 材料在下、5 μm source 在上，保持底部 Robin 冷却。
- 早期功率栅格直接把 block 总功率均摊到 21×21 节点，未复现上游的“20×20 面单元功率 → 四邻单元平均到 21×21 节点”流程。现按真实重叠面积先构造 20×20 cell power，再执行与官方 `convert_interval_to_grid` 等价的转换；x/y 存储约定与上游 Fortran 展平后的序列一致。
- 早期无量纲热源系数把 21 个节点误当成 21 个面单元，得到 1.1025/220.5。按原始 20×20、50 μm 面单元重新推导后，顶面热流系数为 1.0，5 μm 体热源系数为 200.0。
- 物理采样点原先用 `floor(x*21), floor(y*21)` 取功率，既不对应 21 个节点的坐标，也会在单元边界产生跳变。现把 `[x,y]` 连续坐标对 21×21 节点功率做双线性采样，并用非对称节点图回归测试验证轴顺序。
- 重复种子原先反复使用同一训练子集，且预训练模型可能产生几乎相同结果；现每个 data seed 重抽嵌套子集，并在同 seed 下严格配对两种初始化。
- 正式运行前检查原先只看基础 split，现逐个任务验证 `train_size + data_seed`、文件存在性、测试隔离和数据案例 ID。
- 物理梯度诊断原先每个 epoch 只记录首个案例，现汇总所有训练案例并记录负余弦的冲突比例。
- 旧项目数据目录虽有参数表，但温度真值文件数量严重不足，不能作为正式实验；工作站必须重新生成完整真值。

## 仍需由工作站结果决定的结论

以下不是代码完成后就自动成立：PINN 在 10/20 样本是否显著优于随机初始化、哪个物理权重最好、微通道低误差是否仍存在、3D 代理误差是否工程可接受、以及真实加速倍数。工作站产出的结果 JSON 是唯一可用于最终报告的数据；若门禁失败，应报告失败原因而不是挑选好看的案例。

## 主要公开来源

- NVIDIA, *NVIDIA Hopper Architecture In-Depth*: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- NVIDIA H100 product page: https://www.nvidia.com/en-us/data-center/h100/
- AMD Instinct MI300X data sheet: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
- IBM, *Fabrication and performance of 300-mm wafer-scale silicon microchannel cooler*: https://research.ibm.com/publications/fabrication-and-performance-of-300-mm-wafer-scale-silicon-microchannel-cooler
- Sridhar et al., *3D-ICE*, ICCAD 2010: https://www.epfl.ch/labs/esl/wp-content/uploads/2018/12/3D-ICE_ICCAD2010.pdf
- IBM, *3D-ICE: A Compact Thermal Model for Early-Stage Design of Liquid-Cooled ICs*: https://research.ibm.com/publications/3d-ice-a-compact-thermal-model-for-early-stage-design-of-liquid-cooled-ics
- Lu et al., *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence: https://doi.org/10.1038/s42256-021-00302-5
- Wang et al., PINN gradient pathologies: https://arxiv.org/abs/2001.04536
- Wang et al., PINN loss-component convergence imbalance: https://arxiv.org/abs/2007.14527
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*: https://www.sciencedirect.com/science/article/pii/S2666389923001599

## 第二轮上游代码求证（2026-09-06）

### 精确版本与许可证

- 本机 3D-ICE 基线：官方仓库 `esl-epfl/3d-ice`，commit `4953952a1ef6d38807ff307212a6f15e5b2ef935`，describe 为 `4.0-12-g4953952`。官方 README 要求使用成果引用 3D-ICE 4.0、3.0、2014 和 2010 四项工作；代码为 GPL-3.0-or-later。项目若连同模拟器分发，必须保留其 `COPYING`、源码和署名，不能把整个组合笼统标成 MIT。
- 本机 DeepOHeat 基线：官方仓库 `Cadence-Celsius/DeepOHeat`，commit `46ccfe3fd43d99765e427480e7b6e0e16c3dbc70`，上游为 MIT。项目新增训练、模型和部署文件不属于上游原始成果，报告应写成“基于 DeepOHeat 扩展”，不能暗示由上游作者验证。
- 机器可读留痕在 `results/upstream_provenance.json`；工作站须重新执行 `capture_provenance.py`，同时记录模拟器二进制和 PINN checkpoint 的 SHA-256。

### 对上游含义的修正

- 官方 DeepOHeat 的 `2d_power_map` 是“二维功率输入”，并非“只能输出二维温度”。其 `sample_eval_data_single_domain` 会生成三维查询网格，论文/README 也明确支持 2D/3D power map 与三维堆叠几何。
- 本项目历史监督微调把查询坐标固定在 `z=0.5`，因此历史多分支 checkpoint 只能主张固定表面输出；新 `train_3d_revision.py` 才是使用 3D-ICE VTK cell centers 的独立全体积训练轨道。
- 官方 DeepOHeat prototype 对外部 power-map 文件采用 Fortran 顺序展平。我们的 floorplan 栅格第一轴直接定义为 x、第二轴为 y，再用 C 顺序展平，二者在索引序列上等价；此约定必须保留，未来不得随意 `transpose`。

### 3D-ICE 4.0 非均匀 Tmap 风险

官方 4.0 源码 `stack_element_print_thermal_map` 表明：`non-uniform true` 时，Tmap 按内部 cell list 输出一维温度序列，并在模拟器当前目录另写 `xyaxis_<stack id>.txt`；它不是可以直接开平方 reshape 的规则图像。此前通用解析器会静默重排，可能破坏空间对应。

现已加入门禁：普通表面数据加载遇到带 `xyaxis_` 标记的 Tmap 会拒绝；3D 轨道只允许对该 Tmap 做数值范围检查并明确 `spatial_temperature_maps_validated=false`，真正训练必须读取带 cell 坐标的 VTK。20 mm 正式轨道继续使用 VTK，不使用非均匀 Tmap 作为空间监督真值。

### 从同类官方项目得到的启发

- NVIDIA PhysicsNeMo 的 PINN 示例把数据项、方程残差和优化步骤显式分开，并提供热沉/共轭换热案例。它支持我们当前“分别记录数据 Loss、物理 Loss、梯度范数与边界残差”的方向，但不能作为本项目精度证据。
- PhysicsNeMo 与其他成熟 Physics-ML 工程都强调配置、checkpoint、分布式日志和可复现运行。为此新增 provenance JSON；后续正式结果还应保留完整命令矩阵、环境版本、每任务 marker 和日志。
- [Therm-FM](https://github.com/haiyangxin/Therm-FM) 的官方 DAC 2026 实现同时报告反归一化后的 RMSE、最大/平均误差、MAPE/PAPE，并把跨芯片泛化作为单独任务。这启发我们保留逐案例、热点及相对温升指标；但本项目 20 mm 数据仍是同一合成几何族，不能据此声称跨芯片或跨封装泛化。
- [MFIT](https://github.com/AlishKanani/MFIT) 提供多 chiplet、非均匀网格、各向异性材料和稳态/瞬态热模型，并报告与 ANSYS 的验证；[HotSpot HOWTO](https://github.com/danielpalomino/hotspot/blob/master/HOWTO) 明确包含封装及次级散热路径。二者说明当前三层、高 HTC 等效边界缺少封装级次级热路和流体细节，适合代理模型研究，不足以称工程认证数字孪生。

### 第二轮新增的可复核实现结论

- 用实际均匀网格 `case_0_temp.txt` 做非对称热区方向检查时，最高温落在 Tmap 的 `row=y, column=x` 约定；监督目标按 C 顺序展开与训练坐标的 `meshgrid(y,x)` 对齐。该证据只适用于均匀 Tmap；非均匀输出仍必须使用 VTK 坐标。
- 3D 评估不再借用首个训练案例的芯片尺寸。每个 VTK 样本保存自己的 `centers_um`，热点三维距离按本案例真实坐标计算。
- 正式测速强制案例属于冻结 test split，并核对 checkpoint 的任务类型，避免用训练案例或表面/体模型错配生成看似有效的速度数字。
- 3D 轨道的 `250 mW/unit` 与 `50 K/unit` 是预先固定的线性数值缩放：在 20×20 功率面单元、150–500 W 总功率下，平均节点输入约为 1.5–5 个量纲单位。它们没有读取测试集统计量，因而不构成泄露；但也不是自适应归一化或精度保证，正式 JSON 必须保留这两个配置值。
- 五对重复的精确双侧符号翻转检验最小 p 值为 0.0625，先天无法达到 0.05。物理扫描因此从 5 增至 10 个成对 seed（110 个任务）；Student-t 结果保留作描述，Holm 主校正基于不要求正态差值的精确符号翻转 p 值。这里的置信区间单位是“训练运行 seed”，不是独立测试案例的不确定性。
