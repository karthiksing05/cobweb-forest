"""Gradient baselines: an MLP and a B-spline KAN (Liu et al. 2024; the efficient-kan
formulation: each edge is w_b * silu(x) + sum_i c_i B_i(x) with cubic B-splines on a
uniform grid).  Both are trained with Adam on MSE (or cross-entropy) with early stopping
on a validation split."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class KANLayer(nn.Module):
    def __init__(self, d_in, d_out, grid=5, k=3, lo=-1.0, hi=1.0):
        super().__init__()
        self.d_in, self.d_out, self.k = d_in, d_out, k
        h = (hi - lo) / grid
        g = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h + lo
        self.register_buffer("grid", g.expand(d_in, -1).contiguous())
        self.base = nn.Linear(d_in, d_out)
        self.spline = nn.Parameter(torch.randn(d_out, d_in, grid + k) * 0.1 / math.sqrt(d_in))

    def bsplines(self, x):                       # x: (B, d_in) -> (B, d_in, grid+k)
        g = self.grid
        x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            b = ((x - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * b[..., :-1]
                 + (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p]) * b[..., 1:])
        return b

    def forward(self, x):
        out = self.base(Fn.silu(x))
        sp = self.bsplines(x).reshape(x.shape[0], -1)
        return out + sp @ self.spline.reshape(self.d_out, -1).T


class KAN(nn.Module):
    """Hidden layers squash with tanh so every layer's input stays on its spline grid."""
    def __init__(self, widths, grid=5, k=3):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(a, b, grid, k) for a, b in zip(widths[:-1], widths[1:])])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def mlp(d_in, d_out, width=64, depth=3):
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, X, Y, Xv, Yv, task="reg", lr=1e-3, epochs=2000, batch=256, patience=100,
          seed=0, device="cpu"):
    """Adam with early stopping on the validation loss; returns the best model."""
    torch.manual_seed(seed)
    model = model.to(device)
    X, Xv = torch.as_tensor(X, dtype=torch.float32, device=device), torch.as_tensor(Xv, dtype=torch.float32, device=device)
    if task == "reg":
        Y = torch.as_tensor(Y, dtype=torch.float32, device=device).reshape(len(X), -1)
        Yv = torch.as_tensor(Yv, dtype=torch.float32, device=device).reshape(len(Xv), -1)
        loss_fn = Fn.mse_loss
    else:
        Y = torch.as_tensor(Y, dtype=torch.long, device=device)
        Yv = torch.as_tensor(Yv, dtype=torch.long, device=device)
        loss_fn = Fn.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), Yv).item()
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
