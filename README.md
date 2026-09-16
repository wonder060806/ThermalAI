# ThermalAI

Reproducible operator-learning and physics-informed workflows for chip thermal simulation.

> **Research status:** ThermalAI is an actively validated research prototype for learning surrogate models from 3D-ICE simulations. It is not a certified thermal-safety system and has not yet been validated on physical production chips.

## What this repository contains

- DeepONet-based temperature-field prediction code;
- deterministic dataset splits and leakage audits;
- paired small-sample PINN-versus-random experiments;
- physics-loss equation and weight sweeps;
- engineering-scale 20 mm case generation and multilayer VTK validation;
- four-GPU resumable workstation orchestration;
- relative-temperature-rise, hotspot and inference-time evaluation;
- 117 automated regression tests.

Generated datasets, VTK files, model weights and workstation logs are intentionally excluded from Git.

## Scientific question

Given a spatial power map and physical/cooling parameters, ThermalAI studies the mapping

```text
(power distribution, materials, geometry, cooling) -> temperature field
```

The project focuses on whether fast neural surrogates remain accurate under small datasets, realistic chip scale, multilayer 3D outputs and physics-informed training.

## Current evidence

- The paired small-sample experiment completed 70/70 jobs. A weak mean advantage appeared at 10 training cases, but its confidence interval crossed zero; no stable PINN-initialization advantage was observed from 20 to 200 cases.
- A 110-job physics-loss sweep compares a supervised baseline with surface-flux and volumetric-source formulations across five weights and ten paired seeds. It must finish before any general claim that physics constraints help or hurt.
- A representative 20 mm × 20 mm, 150–500 W track produces three distinct thermal layers and passed zero-power, grid-convergence and boundary-sensitivity gates. Formal 3D model training remains a separate validation stage.
- The legacy “microchannel” mode is an equivalent high-HTC boundary model, not a resolved coolant-flow simulation.

Evidence and limitations are tracked in `experiments/report_revision/REPORT_EVIDENCE.md` and `TECHNICAL_AUDIT.md`.

## Quick verification

```bash
git clone https://github.com/wonder060806/ThermalAI.git
cd ThermalAI
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/report_revision/requirements-workstation.txt
.venv/bin/python -m unittest discover -s experiments/report_revision/tests -v
```

Formal simulation and GPU training require external DeepOHeat assets, a separately installed 3D-ICE emulator and a CUDA-enabled Linux workstation. See the revision documentation before running formal experiments.

## Repository layout

```text
scripts/                          legacy multi-mode prototype
src/                              DeepOHeat-derived model components
experiments/report_revision/      revised experiments, tests and documentation
examples/                         small usage examples
docs/                             project background documents
```

## External dependencies and licensing

- [DeepOHeat](https://github.com/Cadence-Celsius/DeepOHeat), MIT License: architecture and selected adapted components.
- [3D-ICE](https://esl.epfl.ch/3d-ice/), GPLv3: external thermal simulator; its source and binaries are not included here.

Original ThermalAI code and documentation are released under the MIT License. Third-party components retain their original copyright and license notices. See `NOTICE.md` and `THIRD_PARTY_LICENSES/`.

## Responsible use

Predictions must not be used as the sole protection against overheating or hardware damage. Validate the model against independent simulations and physical measurements within a documented applicability domain.

## Contributing

Contributions that improve reproducibility, testing, physical correctness or independent validation are welcome. Do not submit private datasets, credentials, generated model weights or material you do not have permission to redistribute.

