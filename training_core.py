import os
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any, Iterable

# DTOFDataset: preprocessing + channel construction 

class DTOFDataset(Dataset):
    """
    DTOF dataset with preprocessing and flexible channel configurations.

    Preprocessing:
        - load CSV
        - crop time axis to [0, crop_t_max]
        - Savitzky-Golay smoothing
        - negative-value clipping
        - per-trace standardisation
        - channel construction:
            * "single"        -> 1 channel, full DTOF
            * "early_mid_late"-> 3 channels (early/mid/late masks)
            * "hybrid_4ch"    -> 4 channels (full + 3 masks)

    Returns:
        signal: (C, T) tensor (C = n. of channels)
        label:  (2,) tensor [mua, mus]
    """

    def __init__(self, csv_path: str, labels: np.ndarray, cfg: dict):
        super().__init__()
        self.cfg = cfg

        df = pd.read_csv(csv_path)

        # Time and DTOFs
        time_full = df.iloc[:, 0].values               # (T_full,)
        dtof_full = df.iloc[:, 1:].values.T            # (N, T_full)
        N, T_full = dtof_full.shape

        # Crop time axis
        t_mask = (time_full >= 0.0) & (time_full <= cfg["crop_t_max"])
        time = time_full[t_mask]                       # (T,)
        dtof = dtof_full[:, t_mask]                    # (N, T)

        # Savitzky–Golay smoothing
        dtof_smooth = savgol_filter(
            dtof,
            cfg["sg_window"],
            cfg["sg_order"],
            axis=1
        )

        # Clip negatives and standardise
        eps = cfg["eps"]
        dtof_smooth[dtof_smooth < 0] = eps

        mean = dtof_smooth.mean(axis=1, keepdims=True)
        std = dtof_smooth.std(axis=1, keepdims=True)
        #dtof_std = (dtof_smooth - mean) / (std + eps)  # (N, T)

        # Build channels
        channels = self.build_channels(time, dtof_smooth, cfg["channel_mode"])
        # channels: (N, C, T)

        self.signals = torch.tensor(channels, dtype=torch.float32)  # (N,C,T)
        self.labels = torch.tensor(labels, dtype=torch.float32)     # (N,2)

        self.N, self.C, self.T = self.signals.shape

    def build_channels(self, t: np.ndarray, dtof: np.ndarray, mode: str) -> np.ndarray:
        """
        Construct channels based on the chosen mode:
            "single"         -> 1 channel, full DTOF
            "early_mid_late" -> 3 masked channels
            "hybrid_4ch"     -> 1 full + 3 masked = 4 channels
        """
        N, T = dtof.shape

        if mode == "single":
            # (N, 1, T)
            return dtof[:, None, :]

        # Define early/mid/late masks within cropped time
        early = ((t >= 0.0) & (t < 0.5)).astype(float)
        mid = ((t >= 0.5) & (t < 4.0)).astype(float)
        late = ((t >= 4.0) & (t <= self.cfg["crop_t_max"])).astype(float)
        masks = np.stack([early, mid, late], axis=0)  # (3, T)

        if mode == "early_mid_late":
            # Multiply each DTOF by masks to get 3 channels
            out = dtof[:, None, :] * masks[None, :, :]  # (N,3,T)
            return out

        if mode == "hybrid_4ch":
            # Channel 1: full DTOF
            ch_full = dtof[:, None, :]                   # (N,1,T)
            ch_bins = dtof[:, None, :] * masks[None, :, :]  # (N,3,T)
            return np.concatenate([ch_full, ch_bins], axis=1)  # (N,4,T)

        raise ValueError(f"Unknown channel_mode: {mode}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]   
    
# CNN Model (Net) with flexible in_channels and FC head 

class Net(nn.Module):
    """
    1D CNN for DTOF-based regression to (μa, μs').

    - Supports variable input channels (C) from CONFIG["in_channels"]
    - Uses three conv blocks with BatchNorm, ReLU, MaxPool
    - Automatically computes flatten dimension
    - FC layers controlled by CONFIG["hidden_fc"]
    """

    def __init__(self, cfg: dict, input_length: int):

        super().__init__()

        C = cfg["in_channels"]

        self.conv1 = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Determine flatten dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, C, input_length)  # (1,C,T)
            feat = self._forward_features(dummy)
            flatten_dim = feat.shape[1]  # (1, flatten_dim)

        # Build FC head from cfg["hidden_fc"]
        fc_layers = []
        last = flatten_dim
        for h in cfg["hidden_fc"]:
            fc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        fc_layers.append(nn.Linear(last, cfg["output_dim"]))
        self.fc = nn.Sequential(*fc_layers)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)  # (batch, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.fc(x)
        return x

# Model Evaluation and visualisation of performance (MAE / RMSE, percentage error)
class ModelEvaluator:
    """
    Evaluate a trained model on a DataLoader and compute MAE/RMSE for (μa, μs'),
    plus percentage error plots vs actual values.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(self, data_loader: DataLoader, cfg: dict):
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in data_loader:
                signals = signals.to(self.device)
                labels  = labels.to(self.device).float()

                preds = self.model(signals)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds  = torch.cat(all_preds,  dim=0)  # (N,2)
        all_labels = torch.cat(all_labels, dim=0)  # (N,2)

        # Basic metrics: MAE and RMSE
        abs_err = torch.abs(all_preds - all_labels)
        sq_err  = (all_preds - all_labels) ** 2

        mae  = abs_err.mean(dim=0)             # (2,)
        rmse = torch.sqrt(sq_err.mean(dim=0))  # (2,)

        # ---------- Percentage error vs Actual ----------
        preds_np  = all_preds.numpy()
        labels_np = all_labels.numpy()

        # Avoid division by very small numbers
        eps = 1e-8
        denom = np.maximum(np.abs(labels_np), eps)  # (N,2)

        pct_error = 100.0 * (preds_np - labels_np) / denom  # signed %
        abs_pct_error = np.abs(pct_error)                   # absolute %

        # Scatter plots: Actual vs % error for μa and μs′
        save_dir = cfg["save_dir"]
        run_name = cfg["run_name"]
        os.makedirs(save_dir, exist_ok=True)

        # error vs true μa plot
        fig_mua = os.path.join(save_dir, f"{run_name}_pct_error_mua.png")
        x = labels_np[:, 0]            # true μa
        y = abs_pct_error[:, 0]        # absolute percentage error

        plt.figure()
        plt.scatter(x, y, s=10, alpha=0.6, label="Absolute % error")

        # Exponential model
        def _exp_model(x, a, b, c):
            return a * np.exp(-b * x) + c

        # Fit curve
        try:
            popt, _ = curve_fit(_exp_model, x, y, p0=[100, 1.0, 0.0], maxfev=5000)
            x_fit = np.linspace(min(x), max(x), 300)
            y_fit = _exp_model(x_fit, *popt)
            plt.plot(
                x_fit, y_fit, "r-", linewidth=2,
                label=f"Fit: a·exp(-b·x) + c\n"
                    f"a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}"
            )
        except Exception as e:
            print("[WARN] μa exponential fit failed:", e)

        plt.xlabel("True μa")
        plt.ylabel("Absolute % error")
        plt.title(f"Percentage error vs Actual μa: {run_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_mua, dpi=150)
        plt.close()

        # error vs true μs′ plot
        fig_mus = os.path.join(save_dir, f"{run_name}_pct_error_mus.png")
        x = labels_np[:, 1]            # true μs'
        y = abs_pct_error[:, 1]        # absolute percentage error

        plt.figure()
        plt.scatter(x, y, s=10, alpha=0.6, label="Absolute % error")

        # Exponential model
        def _exp_model(x, a, b, c):
            return a * np.exp(-b * x) + c

        # Fit curve
        try:
            popt, _ = curve_fit(_exp_model, x, y, p0=[100, 0.5, 0.0], maxfev=5000)
            x_fit = np.linspace(min(x), max(x), 300)
            y_fit = _exp_model(x_fit, *popt)
            plt.plot(
                x_fit, y_fit, "r-", linewidth=2,
                label=f"Fit: a·e^(-b·x) + c\n"
                    f"a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}"
            )
        except Exception as e:
            print("[WARN] μs' exponential fit failed:", e)

        plt.xlabel("True μs'")
        plt.ylabel("Absolute % error")
        plt.title(f"Percentage error vs Actual μs': {run_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_mus, dpi=150)
        plt.close()

        metrics = {
            "MAE": mae.numpy(),          # [MAE_mua, MAE_mus]
            "RMSE": rmse.numpy(),        # [RMSE_mua, RMSE_mus]
            "preds": preds_np,
            "labels": labels_np,
            "pct_error": pct_error,      # signed %
            "abs_pct_error": abs_pct_error,
            "pct_error_plots": {
                "mua": fig_mua,
                "mus": fig_mus,
            }
        }
        return metrics

def get_in_channels(mode: str) -> int:
    """
    Returns number of input channels for the CNN depending on the preprocessing mode.

    mode options:
        "single"          -> 1 channel  (raw DTOF only)
        "early_mid_late"  -> 3 channels (masked temporal bins)
        "hybrid_4ch"      -> 4 channels (raw + 3 temporal bins)
    """
    if mode == "single":
        return 1
    elif mode == "early_mid_late":
        return 3
    elif mode == "hybrid_4ch":
        return 4
    else:
        raise ValueError(f"Unknown channel_mode: {mode}")
