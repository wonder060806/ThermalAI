"""
predict.py — ThermalAI inference API

Usage:
  from predict import ThermalAI
  ai = ThermalAI()
  T = ai.predict(power_map, cooling="liquid")
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)

from multi_branch_model import MultiBranchDeepONet

GRID_SIZE, T_REF, DT = 21, 293.15, 25.0
MODES = ["solid", "air", "liquid", "microchannel"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ThermalAI:
    """AI chip thermal simulator — multi-branch DeepONet"""

    def __init__(self, model_path=None):
        self.model = MultiBranchDeepONet(MODES, device=DEVICE)
        if model_path is None:
            model_path = os.path.join(_PROJECT, "checkpoints", "model_final.pth")
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=DEVICE, weights_only=True)
            self.model.load_state_dict(sd, strict=False)
        else:
            print(f"WARNING: {model_path} not found, using random weights")
        self.model.to(DEVICE)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # Pre-compute coordinate grid
        from src import dataio_utils
        mesh = dataio_utils.fixed_mesh_grid_3d(
            starts=[0, 0], ends=[1, 1], num_intervals=[GRID_SIZE - 1, GRID_SIZE - 1]
        )
        self.coords = torch.tensor(
            np.hstack([mesh, np.ones((len(mesh), 1)) * 0.5]),
            dtype=torch.float32, device=DEVICE,
        )

    def predict(self, power_map, cooling="solid"):
        """
        Predict top-surface temperature field.

        Args:
            power_map: 21x21 numpy array (dimensionless power)
            cooling:   "solid" | "air" | "liquid" | "microchannel"
        Returns:
            21x21 numpy array — temperature in Kelvin
        """
        if cooling not in MODES:
            raise ValueError(f"cooling must be one of {MODES}")

        p = np.asarray(power_map, dtype=np.float32)
        if p.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"power_map must be {GRID_SIZE}x{GRID_SIZE}")

        beta = torch.tensor(p.flatten(), dtype=torch.float32, device=DEVICE)
        beta = beta.unsqueeze(0).repeat(self.coords.shape[0], 1)

        with torch.no_grad():
            u = self.model(self.coords, beta, mode=cooling)
            u = u.cpu().numpy().flatten()

        return (T_REF + DT * u).reshape(GRID_SIZE, GRID_SIZE)

    def predict_raw(self, power_mw_per_cell, cooling="solid"):
        """Input: physical power (mW/cell), auto-normalized"""
        p = np.asarray(power_mw_per_cell, dtype=np.float32) / 0.00625
        return self.predict(p, cooling=cooling)
