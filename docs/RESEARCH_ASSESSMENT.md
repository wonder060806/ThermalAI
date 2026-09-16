# DeepOHeat + 3D-ICE Multi-Fidelity Operator Learning: Research Value Assessment

> **Historical draft — not a final evidence report.** Several claims below (3D output, 1000–300000× speedup, data-scarcity threshold, causal explanation of physics-loss degradation, and four-mode accuracy) were stronger than the available evidence. Use `experiments/report_revision/TECHNICAL_AUDIT.md` and `REPORT_EVIDENCE.md` for the corrected scope.

## 1. What was built

A prototype takes a power distribution map and cooling label as input and predicts a 2D surface temperature field. Historical 5–10 K errors are provisional. A separate revised model now supports full-volume 3D output for one representative high-HTC boundary case; its formal accuracy and speed await workstation results.

**Architecture**: Multi-Branch DeepONet (4 independent branch networks for 4 cooling modes, shared trunk network for spatial encoding). 2.2M parameters. Trained end-to-end with pure supervised learning.

**Supported cooling modes**: solid conduction, air cooling, liquid cold plate, microchannel liquid cooling.

## 2. Novelty

### 2.1 Complete non-dimensionalization derivation

The DeepOHeat paper (DAC 2023) describes experiments in physical units but the code operates entirely in dimensionless space. The normalization rules (temperature, power density, heat transfer coefficient, geometric coordinates) are not documented in the paper. This work systematically reverse-engineered and cross-validated all four mappings from code and paper evidence.

### 2.2 Ablation study revealing limits of physics-informed training

Historical ablations observed C worse than B (5.78 K vs 6.85 K), but this does not establish that physics constraints are generally harmful or identify a unique cause. The revised study scans normalized loss weights, compares source-free and volumetric-source equations, and records gradient conflicts across all cases.

### 2.3 Data-scarcity regime characterization

The 10–200 case sample-efficiency statement is a hypothesis to be tested. The formal matrix uses regenerated complete truth, a fixed grouped test set, independently resampled nested training subsets, and paired PINN/random runs.

### 2.4 Multi-branch architecture for multi-physics regimes

Rather than training one model per cooling mode or a single monolithic model, a multi-branch DeepONet with shared spatial encoding and mode-specific power decoders achieves per-mode specialization without cross-mode interference. Branches are trained independently and merged post-hoc.

## 3. Technical pipeline

```
3D-ICE (FVM simulator)         DeepOHeat (DeepONet framework)
        |                               |
  generate .stk cases          extract model architecture
        |                               |
  FVM solve -> temp.txt        calibrate dimensionless params
        |                               |
        +------- training data ---------+
                        |
              supervised fine-tuning
              (4-mode, 20-fold, 4-GPU)
                        |
              MultiBranchDeepONet
              (model_final.pth)
```

## 4. Experimental Results

| Experiment | Data | Method | Error (K) |
|---|---|---|---|
| Pure PINN baseline | 0 | Physics-only pretraining | 10.72 |
| PINN + supervised (B) | 200 solid | Pretrain then fine-tune | 5.78 |
| PINN + physics + supervised (C) | 200 solid | Hybrid loss fine-tune | 6.85 |
| Random init + supervised (D) | 200 solid | Pure data-driven | **5.40** |
| Unified 4-mode (single model) | 440 total | One-hot mode encoding | 8.68 |
| Multi-branch 4-mode | 440 total | Isolated branches | 5.4-7K |

**Historical observation, not final finding**: random initialization was better in the old 200-case run, while C was worse than B. Dataset completeness, weighting and equation fidelity prevent causal or sample-threshold claims until the revised experiment finishes.

## 5. Publication potential

### 5.1 Strengths

- **Negative result with mechanistic explanation**: C < B is not a failure to report — it is an explainable finding (surface vs volumetric heat source mismatch). The community needs honest ablation results.
- **Data-scarcity quantification**: The PINN-value-vs-datasize curve is a generalizable finding, not hyperparameter tuning.
- **Multi-physics extension**: 4 cooling modes with a single deployable model is more than the original DeepOHeat paper demonstrated.
- **Complete reproducibility**: All scripts, case generators, training configurations are self-contained. Anyone with 3D-ICE + PyTorch can reproduce.

### 5.2 Weaknesses

- The toy material parameters (k=0.1 W/mK) used in training are not representative of real silicon (k=130 W/mK). The microchannel mode uses real silicon but only 80 cases.
- 3D-ICE validation data, while high-fidelity, is still a simulation — no experimental chip measurements.
- The DeepONet architecture is from prior work; architectural novelty is limited to the multi-branch extension.
- Results are on 1mm x 1mm chips only; scaling to realistic multi-die 3D-IC geometries is not demonstrated.
- Only steady-state; transient thermal behavior is not addressed.

### 5.3 Target venues

| Venue | Fit | Reasoning |
|---|---|---|
| DAC (Design Automation Conference) | Strong | Original DeepOHeat venue; EDA audience values thermal simulation |
| DATE | Strong | Similar scope to DAC |
| ISLPED | Moderate | Low-power design, thermal is relevant |
| TCAD | Strong | Full-length paper with detailed experiments |
| NeurIPS/AISTATS ML conferences | Weak | ML novelty is limited; better suited to application-track |

### 5.4 Required additions before submission

1. **Repeat with real silicon parameters**: Re-run solid/air/liquid experiments with k=130 W/mK to show the method works on realistic materials, not just toy values.
2. **Multi-size chips**: Test on 2mm, 5mm chips to demonstrate geometric generalization.
3. **Comparison against pure data-driven baseline from scratch**: Current D baseline uses DeepONet architecture. Add an MLP or CNN baseline to isolate architecture benefit from training method benefit.
4. **Statistical significance**: Run 10-fold CV with confidence intervals, not just 5-fold.
5. **Ablation on trunk sharing**: Quantify how much the shared trunk helps multi-branch vs separate models.

## 6. Data and code

All training data is generated from 3D-ICE via automated scripts. No external dataset dependency. The codebase includes:

- `gen_cases_v3.py`: Solid conduction case generator (200 cases)
- `gen_cases_all.py`: Multi-mode case generator (80 cases/mode for air/liquid/microchannel)
- `heatsink_microchannel.py`: Python-based microchannel thermal model for 3D-ICE pluggable heatsink plugin
- `run_20folds.py`: Multi-GPU training orchestrator (20 tasks across 4 GPUs)
- `multi_branch_model.py`: MultiBranchDeepONet implementation
- `predict.py`: Inference interface

## 7. What this work demonstrates

1. Physics-informed pretraining + data-driven fine-tuning is viable for 3D-IC thermal simulation, but the value of the physics step depends critically on dataset size and physics-model fidelity.
2. A well-designed ablation study (A/B/C/D) reveals more than a hyperparameter sweep: it characterizes *when* each component of a method matters.
3. Open-source tools (3D-ICE + DeepOHeat) can be combined to produce a working AI thermal simulator without access to commercial solvers or proprietary datasets.
