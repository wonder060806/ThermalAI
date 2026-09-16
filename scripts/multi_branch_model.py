"""
multi_branch_model.py — 多分支 DeepONet: 4 个 Branch 各训各的, 共享 Trunk

架构:
  输入 (功率图 + 模式标签) → 路由到对应 Branch → Hadamard × Trunk → 温度

  branch_solid / branch_air / branch_liquid / branch_micro 独立权重
  trunk 共享 (空间编码四种模式通用)
"""
import torch
import torch.nn as nn
import numpy as np
import copy


class FCBlock(nn.Module):
    def __init__(self, in_f, out_f, hidden_f, num_hidden, nonlinearity, device):
        super().__init__()
        self.net = nn.Sequential()
        act = {"silu": nn.SiLU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[nonlinearity]
        prev = in_f
        for i in range(num_hidden):
            self.net.add_module(f"fc{i}", nn.Linear(prev, hidden_f, device=device))
            self.net.add_module(f"act{i}", act)
            prev = hidden_f
        self.net.add_module("out", nn.Linear(prev, out_f, device=device))

    def forward(self, x):
        return self.net(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_f, out_f, freq, std=1, trainable=True):
        super().__init__()
        self.B = nn.Parameter(torch.normal(0, std, (out_f, in_f)) * freq, requires_grad=trainable)

    def forward(self, x):
        proj = 2 * torch.pi * (x @ self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MultiBranchDeepONet(nn.Module):
    """四种冷却模式各有一个 Branch, 共享一个 Trunk"""

    def __init__(self, modes, trunk_in=3, trunk_hidden=128, branch_in=441,
                 branch_hidden=256, inner_prod=128, trunk_layers=3,
                 branch_layers=7, nonlinearity="silu", freq=2*np.pi,
                 std=1, freq_trainable=True, device="cuda"):
        super().__init__()
        self.modes = modes
        self.n_modes = len(modes)
        self.device = device

        # Trunk: 共享空间编码
        self.fourier = FourierFeatures(trunk_in, trunk_hidden // 2, freq, std, freq_trainable)
        trunk_in_f = trunk_hidden  # after Fourier (sin+cos concat)
        self.trunk = FCBlock(trunk_in_f, inner_prod, trunk_hidden, trunk_layers, nonlinearity, device)

        # Branch: 每种模式独立
        self.branches = nn.ModuleDict()
        for mode in modes:
            self.branches[mode] = FCBlock(branch_in, inner_prod, branch_hidden, branch_layers, nonlinearity, device)

        self.to(device)

    def forward(self, coords, beta, mode):
        """mode: str, one of self.modes"""
        trunk_out = self.trunk(self.fourier(coords))  # (N, inner_prod)
        branch_out = self.branches[mode](beta)         # (N, inner_prod)
        return torch.sum(trunk_out * branch_out, dim=-1, keepdim=True)

    def freeze_trunk(self):
        for p in self.trunk.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self):
        for p in self.trunk.parameters():
            p.requires_grad = True

    def freeze_branch(self, mode):
        for p in self.branches[mode].parameters():
            p.requires_grad = False

    def unfreeze_branch(self, mode):
        for p in self.branches[mode].parameters():
            p.requires_grad = True
