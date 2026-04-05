"""
reference_models.py — Non-FM reference models for FM_order comparison

Implements three competitive baselines to challenge the FM_order architecture:

  1. STGFormer   — Spatial-Temporal Graph Transformer (no covariates, no FM)
  2. STAEformer  — Spatio-Temporal Adaptive Embedding Transformer, time-series
                   only (Liu et al., 2023 style; no covariates, no FM)
  3. TFTExoST    — Temporal Fusion Transformer with Exogenous Spatial-Temporal
                   wrapper (covariates: weather, calendar, events)

All models
  - Input  : x  (B, T, S)  normalised time series
  - Output : y  (B, H, S)  multi-step predictions
  - Follow the same train / evaluate interface as FM_spatial_covariates.py

Usage (in main or standalone):
  - Set USE_REFERENCE_MODELS = True  in the __main__ block
  - Set REFERENCE_MODEL to "stgformer", "staef", "tft_exost", or "all"
"""

import os
import sys
import time
import json
import shutil
import tempfile
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def _batch_to_device(batch: dict, key: str, device):
    """Return batch[key] moved to device, or None if key is missing or value is None."""
    v = batch.get(key)
    return v.to(device) if v is not None else None


# ============================================================================
# SHARED BUILDING BLOCKS
# ============================================================================

def _make_ffn(d_model: int, d_ff: int = None, dropout: float = 0.1) -> nn.Sequential:
    d_ff = d_ff or 4 * d_model
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class GRN(nn.Module):
    """
    Gated Residual Network (Lim et al., 2021 — TFT building block).
    Works with any leading batch dimensions: (..., d_in) -> (..., d_out).
    """
    def __init__(self, d_in: int, d_out: int = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_out or d_in
        self.fc1  = nn.Linear(d_in, d_in)
        self.fc2  = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = F.elu(self.fc1(x))
        h   = self.drop(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.norm(out + self.proj(x))


# ============================================================================
# 1. STGFormer — Spatial-Temporal Graph Transformer
# ============================================================================

class _STGFormerLayer(nn.Module):
    """
    One STGFormer layer: interleaved temporal and spatial multi-head attention.

    x : (B, T, S, D)  →  (B, T, S, D)
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.spatial_attn  = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.ffn   = _make_ffn(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape

        # --- Temporal attention: attend over T for each sensor ---
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)      # (B*S, T, D)
        xt_out, _ = self.temporal_attn(xt, xt, xt)
        xt = self.norm1(xt + self.drop(xt_out))
        xt = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)       # (B, T, S, D)

        # --- Spatial attention: attend over S for each timestep ---
        xs = xt.reshape(B * T, S, D)                           # (B*T, S, D)
        xs_out, _ = self.spatial_attn(xs, xs, xs)
        xs = self.norm2(xs + self.drop(xs_out))
        xs = xs.reshape(B, T, S, D)

        # --- Position-wise FFN ---
        out = self.norm3(xs + self.drop(self.ffn(xs)))
        return out


class STGFormer(nn.Module):
    """
    Spatial-Temporal Graph Transformer baseline.

    Interleaved temporal and spatial self-attention with learnable sensor and
    temporal positional embeddings.  No foundation model, no covariates.

    Parameters
    ----------
    num_sensors : int   — number of spatial nodes (S)
    input_len   : int   — past context length (T)
    output_len  : int   — forecast horizon (H)
    hidden_dim  : int   — model width
    num_heads   : int   — attention heads (must divide hidden_dim)
    num_layers  : int   — number of STGFormer layers
    dropout     : float
    """
    use_covariates: bool = False

    def __init__(self, num_sensors: int, input_len: int, output_len: int,
                 hidden_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_len   = input_len
        self.output_len  = output_len
        self.hidden_dim  = hidden_dim

        # Project each scalar sensor value to hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)

        # Learnable positional encodings
        self.sensor_emb  = nn.Parameter(torch.randn(1, 1, num_sensors, hidden_dim) * 0.02)
        self.temporal_pe = nn.Parameter(torch.randn(1, input_len, 1, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            _STGFormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Mean-pool over T then project to H per sensor
        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x : (B, T, S)
        returns : (B, H, S)
        """
        B, T, S = x.shape

        # (B, T, S, 1) → (B, T, S, D)
        z = self.input_proj(x.unsqueeze(-1))
        z = z + self.sensor_emb + self.temporal_pe   # add positional info

        for layer in self.layers:
            z = layer(z)                              # (B, T, S, D)

        z = z.mean(dim=1)                             # (B, S, D) — pool over T
        out = self.pred_head(z).transpose(1, 2)       # (B, H, S)
        return out


# ============================================================================
# 2. STAEformer — Spatio-Temporal Adaptive Embedding Transformer (TS-only)
# ============================================================================

class _STAELayer(nn.Module):
    """
    One STAEformer layer.

    Temporal self-attention followed by spatial attention using the shared
    adaptive sensor embedding as key/value (the STAEformer innovation).

    x            : (B, T, S, D)
    adaptive_emb : (S, D) — shared across all layers
    returns      : (B, T, S, D)
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.spatial_attn  = nn.MultiheadAttention(d_model, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.ffn   = _make_ffn(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adaptive_emb: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape

        # --- Temporal self-attention (per sensor) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt_out, _ = self.temporal_attn(xt, xt, xt)
        xt = self.norm1(xt + self.drop(xt_out))
        xt = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)        # (B, T, S, D)

        # --- Spatial attention: query = current states, key/value = adaptive emb ---
        xs_q  = xt.reshape(B * T, S, D)                         # (B*T, S, D)
        kv    = adaptive_emb.unsqueeze(0).expand(B * T, -1, -1) # (B*T, S, D)
        xs_out, _ = self.spatial_attn(xs_q, kv, kv)
        xs = self.norm2(xs_q + self.drop(xs_out)).reshape(B, T, S, D)

        # --- FFN ---
        out = self.norm3(xs + self.drop(self.ffn(xs)))
        return out


class STAEformer(nn.Module):
    """
    Spatio-Temporal Adaptive Embedding Transformer (time-series only).

    Inspired by STAEformer (Liu et al., 2023).  The key innovation is a set of
    learnable per-sensor adaptive embeddings that serve as key/value in spatial
    attention, implicitly encoding the static graph structure.

    No covariates, no foundation model.

    Parameters
    ----------
    num_sensors : int
    input_len   : int
    output_len  : int
    hidden_dim  : int
    num_heads   : int
    num_layers  : int
    dropout     : float
    """
    use_covariates: bool = False

    def __init__(self, num_sensors: int, input_len: int, output_len: int,
                 hidden_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_len   = input_len
        self.output_len  = output_len
        self.hidden_dim  = hidden_dim

        # --- Adaptive spatial embeddings (the STAEformer core idea) ---
        self.adaptive_emb = nn.Parameter(torch.randn(num_sensors, hidden_dim) * 0.02)

        # Temporal positional encoding
        self.temporal_pe = nn.Parameter(torch.randn(1, input_len, 1, hidden_dim) * 0.02)

        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([
            _STAELayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x : (B, T, S)
        returns : (B, H, S)
        """
        B, T, S = x.shape

        # Embed: (B, T, S, 1) → (B, T, S, D)
        z = self.input_proj(x.unsqueeze(-1))
        # Add temporal PE and adaptive sensor identity
        z = z + self.temporal_pe + self.adaptive_emb.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            z = layer(z, self.adaptive_emb)   # (B, T, S, D)

        z   = z.mean(dim=1)                   # (B, S, D) — pool over T
        out = self.pred_head(z).transpose(1, 2)  # (B, H, S)
        return out


# ============================================================================
# 3. TFTExoST — Temporal Fusion Transformer + ExoST Spatial Wrapper
# ============================================================================

class TFTExoST(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2021) with an Exogenous
    Spatial-Temporal (ExoST) spatial attention wrapper.

    TFT core (channel-independent):
      - Additive covariate fusion with GRN gating
      - LSTM encoder (past) / decoder (future)
      - Multi-head temporal self-attention on decoder states
      - Gated skip connection

    ExoST spatial layer:
      - Cross-sensor multi-head attention applied after per-sensor TFT
      - Enables spatial interaction without a pre-defined graph

    Covariates (all optional via use_* flags):
      - weather  (B, T/H, weather_dim)
      - calendar (B, T/H, date_dim)
      - events   (B, T/H, event_dim)

    Parameters
    ----------
    num_sensors : int
    input_len   : int
    output_len  : int
    date_dim    : int   — calendar feature size
    event_dim   : int   — event/holiday feature size
    weather_dim : int   — weather feature size
    hidden_dim  : int
    num_heads   : int
    lstm_layers : int
    dropout     : float
    """
    use_covariates: bool = True

    def __init__(self, num_sensors: int, input_len: int, output_len: int,
                 date_dim: int = 3, event_dim: int = 0, weather_dim: int = 8,
                 hidden_dim: int = 256, num_heads: int = 8,
                 lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_len   = input_len
        self.output_len  = output_len
        self.hidden_dim  = hidden_dim

        # --- Input projections (each covariate type projected independently) ---
        self.ts_proj = nn.Linear(1, hidden_dim)
        self.date_proj    = nn.Linear(date_dim,    hidden_dim) if date_dim    > 0 else None
        self.event_proj   = nn.Linear(event_dim,   hidden_dim) if event_dim   > 0 else None
        self.weather_proj = nn.Linear(weather_dim, hidden_dim) if weather_dim > 0 else None

        # GRN gating after additive covariate fusion (past and future)
        self.past_grn   = GRN(hidden_dim, dropout=dropout)
        self.future_grn = GRN(hidden_dim, dropout=dropout)

        # Learned positional embedding for future steps (fallback when no future cov)
        self.future_pos_emb = nn.Parameter(torch.randn(1, output_len, hidden_dim) * 0.02)

        # --- LSTM encoder / decoder ---
        lstm_drop = dropout if lstm_layers > 1 else 0.0
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                               batch_first=True, dropout=lstm_drop)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                               batch_first=True, dropout=lstm_drop)

        # --- Temporal self-attention (over H decoder steps) ---
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.attn_grn  = GRN(hidden_dim, dropout=dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # Gated residual (decoder out → gated skip to future input)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_gate = nn.LayerNorm(hidden_dim)

        # --- ExoST spatial attention ---
        self.spatial_attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.spatial_grn   = GRN(hidden_dim, dropout=dropout)
        self.norm_spatial  = nn.LayerNorm(hidden_dim)

        # --- Prediction head ---
        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse_past(self, x_flat: torch.Tensor,
                   date_X: torch.Tensor, event_X: torch.Tensor, weather_X: torch.Tensor,
                   B: int, S: int, T: int,
                   use_date: bool, use_event: bool, use_weather: bool) -> torch.Tensor:
        """
        Project and additively fuse past time series + covariates.

        x_flat : (B*S, T, 1)
        returns : (B*S, T, hidden_dim)
        """
        z = self.ts_proj(x_flat)   # (B*S, T, D)

        if use_date and date_X is not None and self.date_proj is not None:
            # date_X : (B, T, date_dim) → expand to (B*S, T, date_dim)
            d_exp = date_X.unsqueeze(1).expand(B, S, T, -1).reshape(B * S, T, -1)
            z = z + self.date_proj(d_exp)

        if use_event and event_X is not None and self.event_proj is not None:
            e_exp = event_X.unsqueeze(1).expand(B, S, T, -1).reshape(B * S, T, -1)
            z = z + self.event_proj(e_exp)

        if use_weather and weather_X is not None and self.weather_proj is not None:
            w_exp = weather_X.unsqueeze(1).expand(B, S, T, -1).reshape(B * S, T, -1)
            z = z + self.weather_proj(w_exp)

        return self.past_grn(z)   # (B*S, T, D)

    def _fuse_future(self, date_Y: torch.Tensor, event_Y: torch.Tensor, weather_Y: torch.Tensor,
                     B: int, S: int, H: int,
                     use_date: bool, use_event: bool, use_weather: bool) -> torch.Tensor:
        """
        Build future decoder input from position embedding + covariates.

        returns : (B*S, H, hidden_dim)
        """
        z = self.future_pos_emb.expand(B * S, H, self.hidden_dim)   # (B*S, H, D)

        if use_date and date_Y is not None and self.date_proj is not None:
            d_exp = date_Y.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1)
            z = z + self.date_proj(d_exp)

        if use_event and event_Y is not None and self.event_proj is not None:
            e_exp = event_Y.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1)
            z = z + self.event_proj(e_exp)

        if use_weather and weather_Y is not None and self.weather_proj is not None:
            w_exp = weather_Y.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1)
            z = z + self.weather_proj(w_exp)

        return self.future_grn(z)   # (B*S, H, D)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                date_X=None, event_X=None, weather_X=None,
                date_Y=None,  event_Y=None,  weather_Y=None,
                use_date: bool = True, use_event: bool = True, use_weather: bool = True,
                **kwargs) -> torch.Tensor:
        """
        x : (B, T, S)
        returns : (B, H, S)
        """
        B, T, S = x.shape
        H = self.output_len

        # Process each sensor independently: (B*S, T, 1)
        x_flat = x.permute(0, 2, 1).reshape(B * S, T, 1)

        # --- Past: fuse TS + covariates, LSTM encode ---
        z_past   = self._fuse_past(x_flat, date_X, event_X, weather_X,
                                   B, S, T, use_date, use_event, use_weather)
        _, (h_n, c_n) = self.encoder(z_past)   # hidden: (lstm_layers, B*S, D)

        # --- Future: build decoder input, LSTM decode ---
        z_future = self._fuse_future(date_Y, event_Y, weather_Y,
                                     B, S, H, use_date, use_event, use_weather)
        dec_out, _ = self.decoder(z_future, (h_n, c_n))   # (B*S, H, D)

        # --- Temporal self-attention (over H) ---
        attn_out, _ = self.temporal_attn(dec_out, dec_out, dec_out)
        attn_out  = self.attn_grn(attn_out)
        dec_out   = self.norm_attn(dec_out + attn_out)

        # --- Gated skip connection (residual from future input) ---
        gate    = torch.sigmoid(self.gate_proj(dec_out))
        dec_out = self.norm_gate(dec_out * gate + z_future)

        # --- ExoST spatial attention ---
        # (B*S, H, D) → (B, S, H, D) → (B*H, S, D)
        z_sp = dec_out.reshape(B, S, H, self.hidden_dim) \
                      .permute(0, 2, 1, 3) \
                      .reshape(B * H, S, self.hidden_dim)
        sp_out, _ = self.spatial_attn(z_sp, z_sp, z_sp)
        sp_out = self.spatial_grn(sp_out)
        z_sp   = self.norm_spatial(z_sp + sp_out)

        # (B*H, S, D) → (B, H, S, D) → (B, H, S)
        z_final = z_sp.reshape(B, H, S, self.hidden_dim)
        out = self.pred_head(z_final).squeeze(-1)   # (B, H, S)
        return out


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_reference_model(model: nn.Module, train_loader,
                          epochs: int = 50, lr: float = 3e-4,
                          model_name: str = "ref_model",
                          use_date: bool = True,
                          use_event: bool = True,
                          use_weather: bool = True) -> float:
    """
    Train a reference model with masked L1 loss.

    Mirrors train_model() from FM_spatial_covariates.py but without FM calls.
    Covariates are passed only when model.use_covariates is True.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    device    = next(model.parameters()).device

    epoch_losses = []
    start_time   = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x      = batch["x"].to(device)
            y_true = batch["y"].to(device)
            y_mask = batch.get("y_mask")
            if y_mask is not None:
                y_mask = y_mask.to(device)

            kwargs = {}
            if model.use_covariates:
                kwargs = dict(
                    date_X    = _batch_to_device(batch, "x_calendar", device),
                    date_Y    = _batch_to_device(batch, "y_calendar", device),
                    event_X   = _batch_to_device(batch, "x_events",   device),
                    event_Y   = _batch_to_device(batch, "y_events",   device),
                    weather_X = _batch_to_device(batch, "x_weather",  device),
                    weather_Y = _batch_to_device(batch, "y_weather",  device),
                    use_date    = use_date,
                    use_event   = use_event,
                    use_weather = use_weather,
                )

            y_pred = model(x, **kwargs)

            if y_mask is not None:
                loss = (y_mask * (y_pred - y_true).abs()).sum() / (y_mask.sum() + 1e-6)
            else:
                loss = F.l1_loss(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        epoch_losses.append(avg_loss)
        print(f"\rEpoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", end="", flush=True)

    print()
    model.loss_history = epoch_losses
    train_time         = time.time() - start_time
    model.train_time   = train_time

    os.makedirs("results_server/loss_functions", exist_ok=True)
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MAE)")
    plt.title(f"Reference Model: {model_name}")
    plt.tight_layout()
    plt.savefig(f"results_server/loss_functions/loss_{model_name}.png")
    plt.close()

    return train_time


@torch.no_grad()
def evaluate_reference_model(model: nn.Module, test_loader, norm_params: dict,
                              use_date: bool = True,
                              use_event: bool = True,
                              use_weather: bool = True):
    """
    Evaluate a reference model and return de-normalised (y_true, y_pred) arrays.

    Mirrors evaluate_model() from FM_spatial_covariates.py.
    De-normalisation uses norm_params['X_max'] and norm_params['X_min'].

    Returns
    -------
    y_true_denorm : np.ndarray  (N_windows, H, S)
    y_pred_denorm : np.ndarray  (N_windows, H, S)
    """
    model.eval()
    device = next(model.parameters()).device

    y_true_list, y_pred_list = [], []
    start_time = time.time()

    for batch in test_loader:
        x      = batch["x"].to(device)
        y_true = batch["y"].to(device)

        kwargs = {}
        if model.use_covariates:
            kwargs = dict(
                date_X    = _batch_to_device(batch, "x_calendar", device),
                date_Y    = _batch_to_device(batch, "y_calendar", device),
                event_X   = _batch_to_device(batch, "x_events",   device),
                event_Y   = _batch_to_device(batch, "y_events",   device),
                weather_X = _batch_to_device(batch, "x_weather",  device),
                weather_Y = _batch_to_device(batch, "y_weather",  device),
                use_date    = use_date,
                use_event   = use_event,
                use_weather = use_weather,
            )

        y_pred = model(x, **kwargs)
        y_true_list.append(y_true.cpu())
        y_pred_list.append(y_pred.cpu())

    model.eval_time = time.time() - start_time

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    # De-normalise (same convention as FM_spatial_covariates.evaluate_model)
    diff  = norm_params["X_max"] - norm_params["X_min"]
    y_min = norm_params["X_min"]
    return (y_true * diff + y_min).numpy(), (y_pred * diff + y_min).numpy()


# ============================================================================
# RESULTS HELPERS  (mirror of main.py utilities)
# ============================================================================

def _save_results_json(model_name: str, results_dict: dict,
                       results_file: str = "results_server/results.json"):
    """Append / update a single result entry in the shared results JSON."""
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}

    if isinstance(existing, list):          # handle legacy list format
        tmp = {}
        for r in existing:
            if "model_name" in r:
                tmp[r["model_name"]] = r
        existing = tmp

    existing[model_name] = results_dict

    try:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp_f:
            json.dump(existing, tmp_f, indent=2)
            tmp_name = tmp_f.name
        shutil.move(tmp_name, results_file)
    except Exception as e:
        print(f"[warning] Could not save results: {e}")


# ============================================================================
# MAIN — standalone entry point
# ============================================================================

if __name__ == "__main__":

    # ── Control flags ────────────────────────────────────────────────────────
    # Set USE_REFERENCE_MODELS = False to skip this file when called as a
    # script (e.g. from main.py via subprocess or direct run).
    USE_REFERENCE_MODELS = True

    # Choose which reference model(s) to train:
    #   "stgformer"  — STGFormer (no covariates)
    #   "staef"      — STAEformer (no covariates)
    #   "tft_exost"  — TFT-ExoST (with covariates: weather, events, calendar)
    #   "all"        — train all three
    REFERENCE_MODEL = "all"

    # Dataset and training settings (mirror of main.py)
    SETTING = "zurich"          # "zurich" or "nyc"
    EPOCHS  = 1                 # increase for real experiments
    BATCH_SIZE  = 32
    HIDDEN_DIM  = 256
    NUM_HEADS   = 8
    NUM_LAYERS  = 4
    DROPOUT     = 0.1
    LR          = 3e-4

    # Covariate flags (for TFTExoST)
    USE_DATE    = True
    USE_EVENT   = True
    USE_WEATHER = True

    # ── Guard ────────────────────────────────────────────────────────────────
    if not USE_REFERENCE_MODELS:
        print("USE_REFERENCE_MODELS=False — exiting.")
        sys.exit(0)

    # ── Imports ──────────────────────────────────────────────────────────────
    # Ensure project root is on the path regardless of working directory
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from torch.utils.data import DataLoader
    from src.data_loader import TimeSeriesDataset

    def _collate_optional(batch):
        """Collate function that handles None-valued keys (mirrors main.py)."""
        collated = {}
        for k in batch[0].keys():
            vals = [d[k] for d in batch]
            collated[k] = torch.stack(vals, dim=0) if vals[0] is not None else None
        return collated

    def _prepare_data(mode, path, normalization_type, batch_size,
                      exogen_var=True, exo_reduced=True):
        train_ds = TimeSeriesDataset(mode=mode, path=path, train=True,
                                     exogen_var=exogen_var, exo_reduced=exo_reduced,
                                     normalization_type=normalization_type)
        test_ds  = TimeSeriesDataset(mode=mode, path=path, train=False,
                                     exogen_var=exogen_var, exo_reduced=exo_reduced,
                                     normalization_type=normalization_type,
                                     norm_params=train_ds.norm_params)
        kw = dict(batch_size=batch_size, shuffle=False, num_workers=4,
                  pin_memory=True, drop_last=False, collate_fn=_collate_optional)
        return train_ds, test_ds, DataLoader(train_ds, **kw), DataLoader(test_ds, **kw)

    # ── Dataset configuration ────────────────────────────────────────────────
    _settings = {
        "zurich": {"modes": ["ped", "bike", "road"],
                   "path":  "data/train_test"},
        "nyc":    {"modes": ["nyc_bike", "nyc_taxi", "chi_taxi"],
                   "path":  "data/train_test/benchmark"},
    }
    modes = _settings[SETTING]["modes"]
    path  = _settings[SETTING]["path"]

    # ── Model registry ───────────────────────────────────────────────────────
    _ALL_MODELS = {
        "stgformer": {"cls": STGFormer,  "use_covariates": False},
        "staef":     {"cls": STAEformer, "use_covariates": False},
        "tft_exost": {"cls": TFTExoST,   "use_covariates": True},
    }
    MODELS_TO_RUN = (
        _ALL_MODELS if REFERENCE_MODEL == "all"
        else {REFERENCE_MODEL: _ALL_MODELS[REFERENCE_MODEL]}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results_server/predictions", exist_ok=True)
    os.makedirs("results_server/loss_functions", exist_ok=True)

    # ── Main training loop ───────────────────────────────────────────────────
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*70}")

        train_data, test_data, train_loader, test_loader = _prepare_data(
            mode, path, normalization_type="rowwise_minmax", batch_size=BATCH_SIZE
        )

        input_len   = train_data.input_dim
        output_len  = train_data.output_dim
        num_sensors = train_data.num_sensors
        weather_dim = train_data.weather_X.shape[2]
        event_dim   = train_data.events_X.shape[2]
        date_dim    = train_data.calendar_X.shape[2]

        for ref_name, cfg in MODELS_TO_RUN.items():
            print(f"\n--- Reference model: {ref_name} | mode: {mode} ---")

            # Instantiate model
            ModelCls = cfg["cls"]
            if ModelCls is TFTExoST:
                # event_dim=0: the dataset's __getitem__ does not include
                # x_events / y_events keys, so those tensors are never
                # available in batches and event_proj would be allocated
                # but never trained.
                model = TFTExoST(
                    num_sensors=num_sensors,
                    input_len=input_len,
                    output_len=output_len,
                    date_dim=date_dim,
                    event_dim=0,
                    weather_dim=weather_dim,
                    hidden_dim=HIDDEN_DIM,
                    num_heads=NUM_HEADS,
                    dropout=DROPOUT,
                ).to(device)
            else:
                model = ModelCls(
                    num_sensors=num_sensors,
                    input_len=input_len,
                    output_len=output_len,
                    hidden_dim=HIDDEN_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT,
                ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_name = f"ref_{ref_name}_{mode}"
            print(f"Model: {model_name} | Parameters: {n_params:,}")

            # Train
            cov_flags = (dict(use_date=USE_DATE, use_event=USE_EVENT, use_weather=USE_WEATHER)
                         if cfg["use_covariates"] else {})
            train_time = train_reference_model(
                model, train_loader,
                epochs=EPOCHS, lr=LR,
                model_name=model_name,
                **cov_flags,
            )

            # Evaluate
            y_true, y_pred = evaluate_reference_model(
                model, test_loader,
                test_data.norm_params,
                **cov_flags,
            )

            # Metrics (mirrors main.py metric computation)
            y_mask = test_data.Y_mask.numpy()
            mae_masked  = float(np.sum(np.abs(y_true - y_pred) * y_mask) / np.sum(y_mask))
            rmse_masked = float(np.sqrt(np.sum(((y_true - y_pred) ** 2) * y_mask) / np.sum(y_mask)))
            mae_overall  = float(np.mean(np.abs(y_true - y_pred)))
            rmse_overall = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            print(f"MAE (masked): {mae_masked:.4f}, RMSE (masked): {rmse_masked:.4f}"
                  f" | train: {train_time:.1f}s, eval: {model.eval_time:.1f}s")

            # Save predictions
            np.savez(f"results_server/predictions/{model_name}_predictions.npz",
                     Y_pred=y_pred, Y_true=y_true, Y_mask=y_mask)

            # Save to results JSON
            results_entry = {
                "model_name":     model_name,
                "mode":           mode,
                "ref_model_type": ref_name,
                "covariates_used": {
                    "use_date":    USE_DATE    if cfg["use_covariates"] else False,
                    "use_event":   USE_EVENT   if cfg["use_covariates"] else False,
                    "use_weather": USE_WEATHER if cfg["use_covariates"] else False,
                },
                "creation_time":  time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_time":     train_time,
                "inference_time": model.eval_time,
                "metrics": {
                    "MAE_masked":   mae_masked,
                    "MSE_masked":   rmse_masked,
                    "MAE_overall":  mae_overall,
                    "MSE_overall":  rmse_overall,
                },
                "loss_history":         [float(v) for v in model.loss_history],
                "number_of_parameters": n_params,
                "hyperparameters": {
                    "hidden_dim":  HIDDEN_DIM,
                    "num_heads":   NUM_HEADS,
                    "num_layers":  NUM_LAYERS,
                    "epochs":      EPOCHS,
                    "batch_size":  BATCH_SIZE,
                },
            }
            _save_results_json(model_name, results_entry)

    print("\nReference model runs complete.")
