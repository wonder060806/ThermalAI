# ThermalAI Linux 工作站正式实验操作手册

> 目标环境：原生 Linux 工作站，4×RTX 4090，仅通过 SSH / VS Code Remote SSH 管理。  
> 工作区示例：`$HOME/aicooling`。任何一步失败都不要跳过。

远端必须保持下面的结构：

```text
$HOME/aicooling/
├── ThermalAI-full/
├── DeepOHeat/
└── 3d-ice/
```

只复制 `report_revision` 不够，因为正式实验还需要 DeepOHeat 预训练 checkpoint、完整项目数据和 3D-ICE。若 FileZilla 上传后多套了一层 `aicooling/aicooling`，应修正路径后再运行。任何在本机生成、含 Windows 绝对路径的旧 matrix JSON 都不得使用；任务矩阵必须在 Linux 工作站重新生成。

本手册采用“本机完整文件夹直接上传”方式：把本机项目目录上传为 `$HOME/aicooling`，不在服务器上自动移动、合并或删除其他账户目录。不要上传本机虚拟环境；`.venv` 必须在Linux工作站重新创建。

## 0. 设置统一路径

每次重新登录 SSH 后先执行：

```bash
export AICOOLING_ROOT="$HOME/aicooling"
export EXP_ROOT="$AICOOLING_ROOT/ThermalAI-full/experiments/report_revision"
export DEEPOHEAT_ROOT="$AICOOLING_ROOT/DeepOHeat"
export ICE_ROOT="$AICOOLING_ROOT/3d-ice"
export PYTHON_BIN="$EXP_ROOT/.venv/bin/python"
export ICE_EMULATOR="$ICE_ROOT/bin/3D-ICE-Emulator"
cd "$EXP_ROOT"
```

## 1. Linux、GPU、Python与3D-ICE自检

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv tmux

uname -a
nvidia-smi
python3 --version
test -d "$AICOOLING_ROOT/ThermalAI-full"
test -d "$DEEPOHEAT_ROOT"
test -d "$ICE_ROOT"

python3 -m venv "$EXP_ROOT/.venv"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements-workstation.txt

chmod +x ssh_workstation_task.sh
chmod +x "$ICE_EMULATOR"
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
"$PYTHON_BIN" -m unittest discover -s tests -v
```

必须看到4张GPU。如果 PyTorch 显示 CPU 版或 `torch.cuda.device_count()` 不是4，停止，不要生成正式结果。`requirements-workstation.txt` 中的 Torch 范围不保证 pip 自动选择与你驱动匹配的CUDA wheel；若已有管理员配置的CUDA版PyTorch，优先验证现有环境，不要盲目降级。

确认3D-ICE是Linux可执行文件：

```bash
file "$ICE_EMULATOR"
"$ICE_EMULATOR" --help >/dev/null 2>&1 || true
```

`file` 应显示 ELF。若二进制不是Linux ELF或不能运行，在原生Linux中重新构建：

```bash
cd "$ICE_ROOT"
make -j"$(nproc)"
chmod +x "$ICE_EMULATOR"
cd "$EXP_ROOT"
```

最后保存版本与哈希：

```bash
"$PYTHON_BIN" capture_provenance.py --workspace-root "$AICOOLING_ROOT" --output results/upstream_provenance_workstation.json
```

## 2. 纯SSH长任务管理

正式GPU训练使用 `tmux`，SSH断线不会停止。若系统没有tmux：

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

脚本支持 `small`、`physics`、`3d` 三个阶段和 `render/start/status/attach/stop` 操作：

```bash
./ssh_workstation_task.sh render small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh start small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh status small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh attach small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh stop small "$EXP_ROOT" "$PYTHON_BIN"
```

在tmux中按 `Ctrl-b`，再按 `d`，只会脱离界面，不会停止训练。日志保存在 `results/ssh_<阶段>.log`，最终退出码保存在 `results/ssh_<阶段>_status.json`。`stop` 不删除结果、日志或marker；再次 `start` 时，`workstation_runner.py` 会跳过配置哈希一致的已完成任务。

一次只能启动一个 ThermalAI 阶段。脚本会拒绝在 `thermalai-small`、`thermalai-physics` 或 `thermalai-3d` 任一会话仍运行时启动另一个阶段。

## 3. 生成两套正式案例定义

小样本轨道保持旧预训练的1 mm、毫瓦级物理域；工程轨道独立使用20 mm与150–500 W。两者不能合并统计。

```bash
cd "$EXP_ROOT"
"$PYTHON_BIN" generate_sample_efficiency_cases.py --output-dir "$ICE_ROOT/data_sample_efficiency" --num-cases 260 --seed 20260904
"$PYTHON_BIN" generate_realistic_cases.py --output-dir "$ICE_ROOT/data_realistic_20mm" --num-cases 260 --seed 20260904 --chip-size-mm 20 --power-min-w 150 --power-max-w 500
```

此时只有仿真输入，不能开始训练。

## 4. 原生Linux运行小样本3D-ICE真值

`-P 16` 是CPU并发数，可按工作站CPU和内存调整：

```bash
cd "$ICE_ROOT/data_sample_efficiency/stacks"
find . -maxdepth 1 -name 'case_*.stk' -print0 | sort -zV | xargs -0 -n1 -P16 "$ICE_EMULATOR"
cd "$EXP_ROOT"
```

随后冻结测试集并执行数据审计：

```bash
"$PYTHON_BIN" freeze_splits.py --params "$ICE_ROOT/data_sample_efficiency/cases_params.json" --train-sizes 10,20,50,100,200 --test-fraction 0.2 --seed 20260904 --replicates 10 --output results/solid_formal_split.json
"$PYTHON_BIN" audit_dataset.py --data-dir "$ICE_ROOT/data_sample_efficiency" --mode solid --ambient-k 293.15 --split results/solid_formal_split.json --require-complete --output results/solid_full_audit.json
```

审计必须检查完整案例数、跨划分精确重复、近重复和最近邻距离。失败时停止。

## 5. 小样本70任务矩阵、预检与训练

```bash
"$PYTHON_BIN" generate_small_sample_matrix.py --python "$PYTHON_BIN" --split "$EXP_ROOT/results/solid_formal_split.json" --data-dir "$ICE_ROOT/data_sample_efficiency" --checkpoint "$DEEPOHEAT_ROOT/DeepOHeat/2d_power_map/log/pinn_pretrain/checkpoints/model_epoch_2000.pth" --output-dir "$EXP_ROOT/results/small_sample" --matrix-output "$EXP_ROOT/configs/small_sample_workstation.json" --epochs 2000 --learning-rate 0.0001 --power-per-unit-mw 0.00625

"$PYTHON_BIN" workstation_preflight.py --data-dir "$ICE_ROOT/data_sample_efficiency" --matrix configs/small_sample_workstation.json --minimum-cases 250 --expected-gpus 4 --output results/workstation_preflight.json
"$PYTHON_BIN" workstation_runner.py --matrix configs/small_sample_workstation.json --gpus 0,1,2,3 --artifacts workstation_artifacts --dry-run > results/workstation_dry_run.json

./ssh_workstation_task.sh render small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh start small "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh status small "$EXP_ROOT" "$PYTHON_BIN"
```

完成后聚合：

```bash
"$PYTHON_BIN" aggregate_results.py --input-dir results/small_sample --train-sizes 10,20 --seeds 0,1,2,3,4,5,6,7,8,9 --output results/small_sample_10_20_summary.json
"$PYTHON_BIN" aggregate_results.py --input-dir results/small_sample --train-sizes 50,100,200 --seeds 0,1,2,3,4 --output results/small_sample_50_200_summary.json
```

只有状态为 `complete` 才能写报告。差值小于0代表PINN初始化更好；置信区间跨0或Holm校正不显著时，只能写“没有观察到稳定优势”。

## 6. 物理Loss方程与权重110任务

```bash
"$PYTHON_BIN" generate_physics_matrix.py --python "$PYTHON_BIN" --split "$EXP_ROOT/results/solid_formal_split.json" --data-dir "$ICE_ROOT/data_sample_efficiency" --checkpoint "$DEEPOHEAT_ROOT/DeepOHeat/2d_power_map/log/pinn_pretrain/checkpoints/model_epoch_2000.pth" --output-dir "$EXP_ROOT/results/physics_loss" --matrix-output "$EXP_ROOT/configs/physics_workstation.json" --train-size 50 --epochs 2000 --learning-rate 0.0001

"$PYTHON_BIN" workstation_preflight.py --data-dir "$ICE_ROOT/data_sample_efficiency" --matrix configs/physics_workstation.json --minimum-cases 250 --expected-gpus 4 --output results/physics_preflight.json
"$PYTHON_BIN" workstation_runner.py --matrix configs/physics_workstation.json --gpus 0,1,2,3 --artifacts workstation_artifacts_physics --dry-run > results/physics_dry_run.json

./ssh_workstation_task.sh render physics "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh start physics "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh status physics "$EXP_ROOT" "$PYTHON_BIN"
```

完成后：

```bash
"$PYTHON_BIN" aggregate_physics_results.py --input-dir results/physics_loss --output results/physics_loss_summary.json
```

110个任务来自2种有物理含义的方程×5个权重，加纯监督基线，并使用10个成对seed。最终解释同时查看MAE差值、Student-t区间、精确符号翻转检验、Holm校正和梯度冲突，不能笼统写“物理约束有害”。

## 7. 完整微通道等效边界审计

```bash
"$PYTHON_BIN" prepare_microchannel_truth.py --params "$AICOOLING_ROOT/ThermalAI-full/data/microchannel/cases_microchannel_params.json" --output-dir "$ICE_ROOT/data_microchannel_audit"

cd "$ICE_ROOT/data_microchannel_audit/stacks"
find . -maxdepth 1 -name 'case_microchannel_*.stk' -print0 | sort -zV | xargs -0 -n1 -P16 "$ICE_EMULATOR"
cd "$EXP_ROOT"

"$PYTHON_BIN" freeze_splits.py --params "$ICE_ROOT/data_microchannel_audit/cases_microchannel_params.json" --train-sizes 64 --test-fraction 0.2 --seed 20260904 --output results/microchannel_split.json
"$PYTHON_BIN" audit_dataset.py --data-dir "$ICE_ROOT/data_microchannel_audit" --mode microchannel --ambient-k 293 --split results/microchannel_split.json --require-complete --output results/microchannel_full_audit.json
```

审计必须显示80条参数、80张温度图和 `complete=true`。若常量场或总功率基线已经接近神经网络，应判定任务退化，而不是宣传异常低误差。

## 8. 20 mm三维门禁、260例真值与训练

先运行控制案例：

```bash
cd "$ICE_ROOT/data_realistic_20mm/validation/stacks"
printf '%s\0' zero.stk medium.stk fine.stk htc_low.stk htc_high.stk silicon_thin.stk silicon_thick.stk | xargs -0 -n1 -P7 "$ICE_EMULATOR"
cd "$EXP_ROOT"

"$PYTHON_BIN" validate_3d_gates.py --zero "$ICE_ROOT/data_realistic_20mm/validation/zero.vtk" --medium "$ICE_ROOT/data_realistic_20mm/validation/medium.vtk" --fine "$ICE_ROOT/data_realistic_20mm/validation/fine.vtk" --ambient-k 298.15 --zero-tolerance-k 0.05 --convergence-tolerance-k 0.5 --output results/realistic_3d_gates.json
"$PYTHON_BIN" validate_vtk.py --input "$ICE_ROOT/data_realistic_20mm/validation/htc_low.vtk" --output results/sensitivity_htc_low.json
"$PYTHON_BIN" validate_vtk.py --input "$ICE_ROOT/data_realistic_20mm/validation/htc_high.vtk" --output results/sensitivity_htc_high.json
"$PYTHON_BIN" validate_vtk.py --input "$ICE_ROOT/data_realistic_20mm/validation/silicon_thin.vtk" --output results/sensitivity_silicon_thin.json
"$PYTHON_BIN" validate_vtk.py --input "$ICE_ROOT/data_realistic_20mm/validation/silicon_thick.vtk" --output results/sensitivity_silicon_thick.json
```

只有门禁 `status=pass` 才继续：

```bash
cd "$ICE_ROOT/data_realistic_20mm/stacks"
find . -maxdepth 1 -name 'case_*.stk' -print0 | sort -zV | xargs -0 -n1 -P16 "$ICE_EMULATOR"
cd "$EXP_ROOT"

"$PYTHON_BIN" freeze_splits.py --params "$ICE_ROOT/data_realistic_20mm/cases_params.json" --train-sizes 10,20,50,100,200 --test-fraction 0.2 --seed 20260904 --replicates 5 --output results/realistic_3d_split.json
"$PYTHON_BIN" generate_3d_matrix.py --python "$PYTHON_BIN" --split "$EXP_ROOT/results/realistic_3d_split.json" --data-dir "$ICE_ROOT/data_realistic_20mm" --output-dir "$EXP_ROOT/results/realistic_3d" --matrix-output "$EXP_ROOT/configs/realistic_3d_workstation.json" --train-size 200 --epochs 1000 --learning-rate 0.0001
"$PYTHON_BIN" workstation_preflight.py --data-dir "$ICE_ROOT/data_realistic_20mm" --matrix configs/realistic_3d_workstation.json --minimum-cases 250 --expected-gpus 4 --require-3d --output results/realistic_3d_preflight.json
"$PYTHON_BIN" workstation_runner.py --matrix configs/realistic_3d_workstation.json --gpus 0,1,2,3 --artifacts workstation_artifacts_3d --dry-run > results/realistic_3d_dry_run.json

./ssh_workstation_task.sh render 3d "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh start 3d "$EXP_ROOT" "$PYTHON_BIN"
./ssh_workstation_task.sh status 3d "$EXP_ROOT" "$PYTHON_BIN"
```

完成后：

```bash
"$PYTHON_BIN" aggregate_3d_results.py --input-dir results/realistic_3d --output results/realistic_3d_summary.json
```

## 9. 正式推理速度与同案例3D-ICE对比

只在三维训练完成后执行。固定seed 0 checkpoint，测速案例自动取冻结测试集第一例：

```bash
benchmark_case=$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/realistic_3d_split.json"))["split"]["test"][0])')
sim_command=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps([sys.argv[1], sys.argv[2]]))' "$ICE_EMULATOR" "case_${benchmark_case}.stk")

"$PYTHON_BIN" benchmark_revision.py --kind 3d --checkpoint results/realistic_3d/realistic-3d-n200-s0.pth --data-dir "$ICE_ROOT/data_realistic_20mm" --split results/realistic_3d_split.json --case-id "$benchmark_case" --device cuda:0 --warmups 50 --repeats 500 --load-repeats 10 --external-command-json "$sim_command" --external-cwd "$ICE_ROOT/data_realistic_20mm/stacks" --external-repeats 30 --output results/benchmark_4090_3d_test_case.json
```

报告必须区分模型加载、预热推理与模拟器端到端时间，并同时给出中位数和P95；不能沿用未经工作站实测的“数千倍”。

## 10. 结果回传

用 FileZilla/SFTP 将以下内容从工作站下载回本机，保留目录结构：

- `results/`；
- `workstation_artifacts/`；
- `workstation_artifacts_physics/`；
- `workstation_artifacts_3d/`；
- `configs/*_workstation.json`；
- 工作站实际使用的 split JSON、审计 JSON、日志和 provenance JSON。

任一汇总不是 `complete`、任一preflight失败、三维门禁失败，或微通道不足80张真值，都不能宣布对应老师问题已经由实验解决。
