import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline


class Chronos2:
    def __init__(self):
        self.name = "Chronos-2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.device)
        self.pipeline.model.eval()
    
    @torch.no_grad()
    def predict(self, x, horizon):
        """
        x: torch.Tensor of shape (B, T, S)
        horizon: int
        returns: torch.Tensor of shape (B, H, S)
        """
        B, T, S = x.shape
        x = x.to("cpu")
        # Reshape to (B*S, T) - process each sensor independently
        x_flat = x.permute(0, 2, 1).reshape(B * S, T)
        # Convert to list of 1D tensors (required by Chronos)
        series_list = list(x_flat)
        # Batched Chronos call
        forecasts = self.pipeline.predict(
            inputs=series_list,
            prediction_length=horizon
        )
        # Stack and reshape back
        preds = torch.stack(
            [torch.as_tensor(f).mean(dim=(0, 1)) for f in forecasts]
        )  # (B*S, H)
        preds = preds[:, :horizon]
        # Reshape to (B, S, H) → (B, H, S)
        preds = preds.view(B, S, horizon).permute(0, 2, 1)
        return preds.to(self.device, non_blocking=True)
    
    @torch.no_grad()
    def encode_hidden(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        x: (B, T, S)
        returns: (B, D, S)  (D = hidden size)
        """
        B, T, S = x.shape

        # IMPORTANT: keep series on CPU for pipeline/DataLoader pin_memory
        x_cpu = x.detach().to("cpu")

        x_flat = x_cpu.permute(0, 2, 1).reshape(B * S, T)
        series_list = [ts for ts in x_flat]  # each: CPU tensor (T,)

        emb, _ = self.pipeline.embed(series_list)  # list of (L, D) or tensor
        
        # emb is list[(L_i, D)]  or list[(L, D)]
        emb = torch.stack(emb)        # -> (B*S, L, D)

        # Pool over token axis
        if pool == "mean":
            z = emb.mean(dim=1)
        elif pool == "last":
            z = emb[:, -1, :]
        else:
            raise ValueError("pool must be 'mean' or 'last'")

        # Back to (B, D, S)
        z = z.view(B, S, -1).permute(0, 2, 1).contiguous()
        z = torch.nan_to_num(z, nan=0.0)
        return z.to(self.device, non_blocking=True)
    

class MomentWrapper:
    """
    Differentiable wrapper for MOMENT-1-large foundation model.

    Unlike Chronos-2 (which uses discrete tokenization), MOMENT operates on
    continuous patch embeddings and supports backpropagation through its
    forecasting head. This enables Stage 1 / input-phase training to actually
    update the InputModule via gradients.

    Interface matches Chronos2:  predict(x, horizon) -> (B, H, S)
                                 encode_hidden(x)     -> (B, D, S)
    where D = 1024 (MOMENT-1-large T5 hidden dim).
    """

    N_PATCHES = 64    # 512-step context / patch_len 8
    D_MODEL   = 1024  # T5-Large hidden size

    def __init__(self, input_len=96, output_len=18):
        self.name = "MOMENT"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_len  = input_len
        self.output_len = output_len
        self.seq_len    = 512          # MOMENT's fixed context window

        from momentfm import MOMENTPipeline
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name":        "forecasting",
                "forecast_horizon": output_len,
                "seq_len":          self.seq_len,
            },
        )
        self.model.init()
        self.model.to(self.device)

        # Freeze transformer backbone; keep the linear forecasting head trainable
        # so that Stage-2 / prediction-module training can update it.
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)

        # Persistent hook: captures the encoder output (head input) on every
        # forward pass so that encode_hidden() can reuse it without a second
        # MOMENT forward call.
        self._last_enc_out = None
        self._hook_handle   = self.model.head.linear.register_forward_hook(
            self._capture_enc_hook
        )

    def _capture_enc_hook(self, module, inp, out):
        """Store head input (= pooled encoder hidden states) during forward."""
        # inp[0]: (B, S, N_PATCHES * D_MODEL)  — keeps grad_fn
        self._last_enc_out = inp[0]

    def _prepare_input(self, x):
        """Pad (B, T, S) to MOMENT's expected (B, S, seq_len) with mask."""
        B, T, S = x.shape
        x_bst   = x.permute(0, 2, 1)                                    # (B, S, T)
        x_pad   = torch.zeros(B, S, self.seq_len, device=x.device, dtype=x.dtype)
        x_pad[:, :, -T:] = x_bst                                        # right-align
        mask    = torch.zeros(B, self.seq_len, device=x.device)
        mask[:, -T:] = 1                                                 # 1 = real
        return x_pad, mask

    def predict(self, x, horizon):
        """
        Differentiable forward pass through MOMENT.

        x: (B, T, S)  — same convention as Chronos2
        returns: (B, H, S)  — gradients flow through the forecasting head
        """
        x_pad, mask = self._prepare_input(x)
        out = self.model(x_enc=x_pad, input_mask=mask, forecast_horizon=horizon)
        # out.forecast: (B, S, H) → transpose to (B, H, S)
        return out.forecast.permute(0, 2, 1).to(self.device)

    def encode_hidden(self, x, pool: str = "mean"):
        """
        Return encoder hidden states.

        If called right after predict() on the same batch, reuses the cached
        hook output — no second MOMENT forward pass.

        x: (B, T, S)
        returns: (B, D, S)  where D = 1024
        """
        if self._last_enc_out is None:
            # Fallback: run a forward pass to populate the cache
            x_pad, mask = self._prepare_input(x)
            _ = self.model(x_enc=x_pad, input_mask=mask, forecast_horizon=self.output_len)

        h = self._last_enc_out                     # (B, S, N_PATCHES * D_MODEL)
        B, S, _ = h.shape
        h = h.view(B, S, self.N_PATCHES, self.D_MODEL)   # (B, S, 64, 1024)

        if pool == "mean":
            z = h.mean(dim=2)                      # (B, S, 1024)
        elif pool == "last":
            z = h[:, :, -1, :]                     # (B, S, 1024)
        else:
            raise ValueError("pool must be 'mean' or 'last'")

        return z.permute(0, 2, 1).to(self.device)  # (B, 1024, S)
    
    
"""
foundation_models_new.py
FM_exo wrappers for TimesFM-2.5, MOIRAI-2, and FlowState.

Install:
    TimesFM  : pip install timesfm
    MOIRAI-2 : pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
    FlowState: pip install tsfm_public
"""



def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# TimesFM-2.5  (Google)
# D = 1280  (model_dims, fixed for all public checkpoints)
# Differentiable: YES — continuous patch embeddings
# Multivariate: channel-independent (loop over S channels)
# ──────────────────────────────────────────────────────────────────────────────

class TimesFM25Wrapper:
    D = 1280
    _PATCH = 32   # input_patch_len, fixed for all public checkpoints

    def __init__(self, device, freq=0):
        self.name   = "TimesFM-2.5-200M"
        self.device = device
        self._freq  = freq
        self._cache = None
        self._pass  = 0   # hook fires twice per call; we want pass 0 only
        
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../timesfm/src'))
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        tfm = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        
        self._model = tfm.model.to(device)
        _freeze(self._model)

        # Hook the last transformer layer to cache context hidden states.
        # TimesFM-2.5 uses stacked_xf (not stacked_transformer_layer).
        # Hook fires once per decode() call (context prefill only for horizon<128).
        last_layer = self._model.stacked_xf[-1]
        last_layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        # stacked_xf layers return (output_embeddings, cache)
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        if self._pass == 0:
            self._cache = hidden   # (B*S, n_ctx_patches, D)
        self._pass += 1

    def predict(self, x, horizon):
        B, T, S = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * S, T).to(self.device)
        # Pad T to a multiple of patch size (32) so reshape works inside decode()
        p = self._PATCH
        pad_len = (-T) % p
        if pad_len:
            x_ci = torch.cat([torch.zeros(B * S, pad_len, device=self.device), x_ci], dim=1)
        masks = torch.zeros_like(x_ci, dtype=torch.bool)
        if pad_len:
            masks[:, :pad_len] = True  # mark front-padded positions as masked
        self._pass = 0
        # decode() handles internal normalization, patching, and AR generation
        renormed_out, _, ar_out = self._model.decode(horizon=horizon, inputs=x_ci, masks=masks)
        # renormed_out: (B*S, n_patches, output_len=128, q=10) — last patch, median quantile
        preds = renormed_out[:, -1, :horizon, self._model.aridx]  # (B*S, H)
        return preds.reshape(B, S, horizon).permute(0, 2, 1).contiguous()

    def encode_hidden(self, x, pool="mean"):
        B, T, S = x.shape
        if self._cache is None:
            self.predict(x, horizon=1)
        n_ctx = math.ceil(T / self._PATCH)
        enc = self._cache[:, :n_ctx, :]          # (B*S, n_ctx_patches, D)
        enc = enc.mean(1) if pool == "mean" else enc[:, -1, :]   # (B*S, D)
        self._cache = None
        return enc.reshape(B, S, self.D).permute(0, 2, 1).contiguous()  # (B, D, S)


# ──────────────────────────────────────────────────────────────────────────────
# MOIRAI-2  (Salesforce) — moirai-2.0-R-small is public
# D: small=384  (d_model from HF config)
# patch_size: small=16  (moirai-2 reduced from 32 → 16)
# Note: from_pretrained() API mismatch in uni2ts 2.0.0; load manually.
# Differentiable: YES — continuous patch embeddings
# Channel-independent: reshapes (B,T,S) → (B*S, T, 1)
# ──────────────────────────────────────────────────────────────────────────────

_MOIRAI2_D  = {"small": 384, "base": 768, "large": 1024}
_MOIRAI2_PS = {"small": 16,  "base": 16,  "large": 16}

class MOIRAI2Wrapper:
    _SAMPLES = 100

    def __init__(self, device, size="small"):
        self.name    = f"MOIRAI-2-{size}"
        self.device  = device
        self.D       = _MOIRAI2_D[size]
        self._patch  = _MOIRAI2_PS[size]
        self._cache  = None
        self._forecast = None
        self._pred_len = None
        self._ctx      = None

        from uni2ts.model.moirai import MoiraiModule
        from uni2ts.distribution import StudentTOutput
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import json as _json

        _repo = f"Salesforce/moirai-2.0-R-{size}"
        _cfg = _json.load(open(hf_hub_download(_repo, "config.json")))
        self._module = MoiraiModule(
            distr_output=StudentTOutput(),
            d_model=_cfg["d_model"],
            num_layers=_cfg["num_layers"],
            patch_sizes=(_cfg["patch_size"],),
            max_seq_len=_cfg["max_seq_len"],
            attn_dropout_p=_cfg.get("attn_dropout_p", 0),
            dropout_p=_cfg.get("dropout_p", 0),
            scaling=_cfg.get("scaling", True),
        )
        self._module.load_state_dict(
            load_file(hf_hub_download(_repo, "model.safetensors")), strict=False
        )
        self._module = self._module.to(device)
        _freeze(self._module)

        # Hook the last encoder transformer layer.
        self._module.encoder.layers[-1].register_forward_hook(
            lambda _m, _i, out: setattr(
                self, "_cache", out[0] if isinstance(out, (tuple, list)) else out
            )
        )

    def _get_forecast(self, horizon, T):
        from uni2ts.model.moirai import MoiraiForecast
        if self._forecast is None or self._pred_len != horizon or self._ctx != T:
            self._forecast = MoiraiForecast(
                module=self._module,
                prediction_length=horizon,
                context_length=T,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                patch_size=self._patch,
                num_samples=self._SAMPLES,
            ).to(self.device)
            self._pred_len, self._ctx = horizon, T
        return self._forecast

    def predict(self, x, horizon):
        B, T, S = x.shape
        x_ci = x.permute(0, 2, 1).reshape(B * S, T, 1).to(self.device)
        obs  = torch.ones(B * S, T, 1, dtype=torch.bool, device=self.device)
        pad  = torch.zeros(B * S, T, dtype=torch.bool, device=self.device)
        fc   = self._get_forecast(horizon, T)
        with torch.enable_grad():
            out = fc(past_target=x_ci, past_observed_target=obs, past_is_pad=pad,
                     num_samples=self._SAMPLES)
        # out: (B*S, samples, H, 1) → mean over samples → (B*S, H)
        preds = out.mean(1).squeeze(-1)
        return preds.reshape(B, S, horizon).permute(0, 2, 1).contiguous()

    def encode_hidden(self, x, pool="mean"):
        B, T, S = x.shape
        if self._cache is None:
            self.predict(x, horizon=1)
        n_ctx = math.ceil(T / self._patch)
        enc   = self._cache[:, :n_ctx, :]          # (B*S, n_ctx, D)
        enc   = enc.mean(1) if pool == "mean" else enc[:, -1, :]
        self._cache = None
        return enc.reshape(B, S, self.D).permute(0, 2, 1).contiguous()  # (B, D, S)


# ──────────────────────────────────────────────────────────────────────────────
# FlowState  (IBM Research)
# D = 256  (coefficient-space dim; 9.1M param SSM)
# Differentiable: YES — linear SSM state updates; algebraic basis functions
# Multivariate: channel-independent
# ──────────────────────────────────────────────────────────────────────────────

class FlowStateWrapper:
    # backbone_hidden_state from FlowStateForPrediction has shape (1, B*S, 512).
    # encoder_state_dim=512 is the SSM hidden dimension (not decoder_dim=256).
    D = 512

    def __init__(self, device,
        hf_repo="ibm-granite/granite-timeseries-flowstate-r1"):
        self.name   = "FlowState-r1"
        self.device = device
        self._cache = None   # caches backbone_hidden_state[0] between predict/encode_hidden

        from tsfm_public.models.flowstate import FlowStateForPrediction
        self._model = FlowStateForPrediction.from_pretrained(hf_repo)
        self._model = self._model.to(device)
        _freeze(self._model)

    def predict(self, x, horizon):
        B, T, S = x.shape
        # FlowState expects (B*S, seq_len, 1) — channel-independent
        x_ci = x.permute(0, 2, 1).reshape(B * S, T, 1).to(self.device)
        with torch.enable_grad():
            out = self._model(past_values=x_ci, prediction_length=horizon)
        # Cache backbone hidden state (1, B*S, 512) → (B*S, 512)
        if out.backbone_hidden_state is not None:
            self._cache = out.backbone_hidden_state[0]   # (B*S, 512)
        # prediction_outputs: (B*S, H, 1) → squeeze → (B*S, H)
        preds = out.prediction_outputs.squeeze(-1)
        return preds.reshape(B, S, horizon).permute(0, 2, 1).contiguous()

    def encode_hidden(self, x, pool="mean"):
        B, T, S = x.shape
        if self._cache is None:
            self.predict(x, horizon=1)
        enc = self._cache   # (B*S, 512)
        self._cache = None
        return enc.reshape(B, S, self.D).permute(0, 2, 1).contiguous()  # (B, 512, S)


# ============================================================================
# COVARIATE ENCODERS
# ============================================================================

class DateEncoder(nn.Module):
    """Encode periodic date features: hour, day-of-week, month"""
    def __init__(self, date_dim=3, d_model=128):
        super().__init__()
        # Categorical embeddings
        self.hour_emb = nn.Embedding(24, 16)
        self.dow_emb = nn.Embedding(7, 8)
        self.month_emb = nn.Embedding(12, 8)
        
        # Fourier features for periodicity
        self.register_buffer('hour_freq', torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer('dow_freq', torch.tensor([1.0]))
        self.register_buffer('month_freq', torch.tensor([1.0]))
        
        # Output projection
        fourier_dim = (len(self.hour_freq) + len(self.dow_freq) + len(self.month_freq)) * 2
        self.proj = nn.Sequential(
            nn.Linear(16 + 8 + 8 + fourier_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
    
    def forward(self, date_features):
        """
        date_features: (B, L, K_date) where K_date=3: [hour, dow, month]
        returns: (B, L, d_model)
        """
        B, L, K = date_features.shape
        
        # Extract features (assume normalized 0-1, convert to indices)
        hour = (date_features[:, :, 0] * 23).long()  # (B, L)
        dow = (date_features[:, :, 1] * 6).long()  # (B, L)
        month = (date_features[:, :, 2] * 11).long()  # (B, L)
        
        # Embeddings
        h_emb = self.hour_emb(hour)  # (B, L, 16)
        d_emb = self.dow_emb(dow)  # (B, L, 8)
        m_emb = self.month_emb(month)  # (B, L, 8)
        
        # Fourier features
        hour_norm = date_features[:, :, 0:1]  # (B, L, 1)
        dow_norm = date_features[:, :, 1:2]  # (B, L, 1)
        month_norm = date_features[:, :, 2:3]  # (B, L, 1)
        
        hour_rad = 2 * math.pi * hour_norm
        dow_rad = 2 * math.pi * dow_norm
        month_rad = 2 * math.pi * month_norm
        
        hour_fourier = torch.cat([
            torch.sin(hour_rad * f) for f in self.hour_freq
        ] + [
            torch.cos(hour_rad * f) for f in self.hour_freq
        ], dim=-1)  # (B, L, 6)
        
        dow_fourier = torch.cat([
            torch.sin(dow_rad * f) for f in self.dow_freq
        ] + [
            torch.cos(dow_rad * f) for f in self.dow_freq
        ], dim=-1)  # (B, L, 2)
        
        month_fourier = torch.cat([
            torch.sin(month_rad * f) for f in self.month_freq
        ] + [
            torch.cos(month_rad * f) for f in self.month_freq
        ], dim=-1)  # (B, L, 2)
        
        # Concatenate all
        all_features = torch.cat([
            h_emb, d_emb, m_emb,
            hour_fourier, dow_fourier, month_fourier
        ], dim=-1)  # (B, L, 16+8+8+6+2+2)
        
        return self.proj(all_features)  # (B, L, d_model)


class EventEncoder(nn.Module):
    """Encode sparse event features"""
    def __init__(self, event_dim, d_model=128):
        super().__init__()
        self.event_dim = event_dim
        if event_dim > 0:
            self.event_proj = nn.Linear(event_dim, d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    
    def forward(self, event_features):
        """
        event_features: (B, L, K_event) or None
        returns: (B, L, d_model)
        """
        if self.event_dim == 0 or event_features is None:
            B = event_features.shape[0] if event_features is not None else 1
            L = event_features.shape[1] if event_features is not None else 1
            device = event_features.device if event_features is not None else 'cpu'
            return torch.zeros(B, L, self.d_model, device=device)
        
        B, L, K = event_features.shape
        # Already in (B, L, K) format
        x = self.event_proj(event_features)  # (B, L, d_model)
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm(x + attn_out)
        return x  # (B, L, d_model)


class WeatherEncoder(nn.Module):
    """Encode continuous weather features"""
    def __init__(self, weather_dim, d_model=128):
        super().__init__()
        # Use MLP instead of Conv1d since input is (B, L, K)
        self.proj = nn.Sequential(
            nn.Linear(weather_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, weather_features):
        """
        weather_features: (B, L, K_weather)
        returns: (B, L, d_model)
        """
        return self.proj(weather_features)  # (B, L, d_model)


# ============================================================================
# ATTENTION AND GRAPH MODULES
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention for sensor-covariate fusion"""
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, queries, context):
        """
        queries: (B, N, D)
        context: (B, L, D)
        returns: (B, N, D)
        """
        attn_out, _ = self.cross_attn(queries, context, context)
        return self.norm(queries + self.scale * attn_out)


class GraphAttentionLayer(nn.Module):
    """Graph attention for spatial dependencies"""
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x, adj=None):
        """
        x: (B, S, T, D)
        adj: (S, S) optional
        returns: (B, S, T, D)
        """
        B, S, T, D = x.shape
        
        # Reshape: (B*T, S, D)
        x_flat = x.transpose(1, 2).reshape(B * T, S, D)
        
        # Multi-head attention
        Q = self.q_proj(x_flat).view(B * T, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x_flat).view(B * T, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x_flat).view(B * T, S, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if adj is not None:
            mask = (adj == 0).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B * T, S, D)
        out = self.out_proj(out)
        
        x_flat = self.norm(x_flat + out)
        x_flat = x_flat + self.ffn(x_flat)
        
        return x_flat.view(B, T, S, D).transpose(1, 2)


# ============================================================================
# MAIN MODEL
# ============================================================================
"""
class SpatialTemporalFM(nn.Module):
    def __init__(
        self,
        fm_encoder,
        num_sensors,
        input_len,
        output_len,
        date_dim=3,  # hour, dow, month
        event_dim=0,
        weather_dim=5,
        fm_hidden_dim=6144,
        hidden_dim=256,
        graph_layers=2,
        use_road_network=False,
        road_adj=None,
        device=None
    ):
        super().__init__()
        self.device = device
        self.fm_encoder = fm_encoder
        self.num_sensors = num_sensors
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.fm_hidden_dim = fm_hidden_dim
        
        # Freeze FM
        for param in self.fm_encoder.pipeline.model.parameters():
            param.requires_grad = False
        
        # Covariate encoders
        self.date_encoder = DateEncoder(date_dim, d_model=128)
        self.event_encoder = EventEncoder(event_dim, d_model=128)
        self.weather_encoder = WeatherEncoder(weather_dim, d_model=128)
        self.covariate_proj = nn.Linear(128, hidden_dim)
        
        # FM adapter: (B, D, S) -> (B, S, hidden_dim)
        self.fm_adapter = nn.Sequential(
            nn.Linear(fm_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.spatial_pos = nn.Parameter(torch.randn(1, num_sensors, 1, hidden_dim))
        
        # Past covariate fusion
        self.past_date_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        self.past_event_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        self.past_weather_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        
        # Spatial graph layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads=8) for _ in range(graph_layers)
        ])
        
        # Road network
        self.use_road_network = use_road_network
        if use_road_network and road_adj is not None:
            self.register_buffer('road_adj', road_adj)
        else:
            self.road_adj = None
        
        # Future projection
        self.future_proj = nn.Linear(hidden_dim, output_len * hidden_dim)
        
        # Future covariate fusion
        self.future_date_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        self.future_event_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        self.future_weather_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    # ============================================================================
    # UPDATE TO FORWARD METHOD IN SpatialTemporalFM CLASS
    # ============================================================================
    # Replace the forward method in your SpatialTemporalFM class with this version

    def forward(self, x, date_X, event_X, weather_X, date_Y, event_Y, weather_Y, 
                use_date=True, use_event=True, use_weather=True, 
                use_past_date=None, use_past_event=None, use_past_weather=None,
                use_future_date=None, use_future_event=None, use_future_weather=None,
                raw_fm=False):
        B, T, S = x.shape
        H = self.output_len
        
        # ========================================================================
        # RESOLVE FLAGS: Use specific past/future flags if provided, else use general
        # ========================================================================
        use_past_date = use_date if use_past_date is None else use_past_date
        use_past_event = use_event if use_past_event is None else use_past_event
        use_past_weather = use_weather if use_past_weather is None else use_past_weather
        use_future_date = use_date if use_future_date is None else use_future_date
        use_future_event = use_event if use_future_event is None else use_future_event
        use_future_weather = use_weather if use_future_weather is None else use_future_weather
        
        # === Get FM hidden states and predictions ===
        with torch.no_grad():
            fm_preds = self.fm_encoder.predict(x, H)  # (B, H, S)
            fm_hidden = self.fm_encoder.encode_hidden(x, pool='mean')  # (B, D, S)
            
        if raw_fm:
            return fm_preds
        
        # === Project FM hidden states ===
        fm_hidden = fm_hidden.transpose(1, 2)  # (B, S, 6144)
        Z = self.fm_adapter(fm_hidden)  # (B, S, hidden_dim)
        Z = Z.unsqueeze(2)  # (B, S, 1, hidden_dim)
        Z = Z + self.spatial_pos
        
        # ========================================================================
        # PAST COVARIATE PROCESSING (using use_past_* flags)
        # ========================================================================
        if use_past_date:
            date_past = self.date_encoder(date_X)
            date_past = date_past.to(self.device, non_blocking=True)
            date_past = self.covariate_proj(date_past)
        else:
            date_past = None
            
        if use_past_event and event_X is not None:
            event_past = self.event_encoder(event_X)
            event_past = event_past.to(self.device, non_blocking=True)
            event_past = self.covariate_proj(event_past)
        else:
            event_past = None
            
        if use_past_weather:
            weather_past = self.weather_encoder(weather_X)
            weather_past = weather_past.to(self.device, non_blocking=True)
            weather_past = self.covariate_proj(weather_past)
        else:
            weather_past = None
        
        # === Past covariate fusion ===
        Z_query = Z.squeeze(2).unsqueeze(1).reshape(B * S, 1, self.hidden_dim)
        
        if use_past_date and date_past is not None:
            date_past_exp = date_past.unsqueeze(1).expand(B, S, T, self.hidden_dim).reshape(B * S, T, self.hidden_dim)
            Z_query = self.past_date_fusion(Z_query, date_past_exp)
            
        if use_past_event and event_past is not None:
            event_past_exp = event_past.unsqueeze(1).expand(B, S, T, self.hidden_dim).reshape(B * S, T, self.hidden_dim)
            Z_query = self.past_event_fusion(Z_query, event_past_exp)
            
        if use_past_weather and weather_past is not None:
            weather_past_exp = weather_past.unsqueeze(1).expand(B, S, T, self.hidden_dim).reshape(B * S, T, self.hidden_dim)
            Z_query = self.past_weather_fusion(Z_query, weather_past_exp)
        
        Z = Z_query.view(B, S, 1, self.hidden_dim)
        
        # === Spatial graph layers ===
        for graph_layer in self.graph_layers:
            Z = graph_layer(Z, adj=self.road_adj)
        
        # === Project to future ===
        Z_spatial = Z.squeeze(2)
        Z_future_flat = self.future_proj(Z_spatial)
        Z_future = Z_future_flat.view(B, S, H, self.hidden_dim)
        
        # ========================================================================
        # FUTURE COVARIATE PROCESSING (using use_future_* flags)
        # ========================================================================
        if use_future_date:
            date_future = self.date_encoder(date_Y)
            date_future = date_future.to(self.device, non_blocking=True)
            date_future = self.covariate_proj(date_future)
        else:
            date_future = None
            
        if use_future_event and event_Y is not None:
            event_future = self.event_encoder(event_Y)
            event_future = event_future.to(self.device, non_blocking=True)
            event_future = self.covariate_proj(event_future)
        else:
            event_future = None
            
        if use_future_weather:
            weather_future = self.weather_encoder(weather_Y)
            weather_future = weather_future.to(self.device, non_blocking=True)
            weather_future = self.covariate_proj(weather_future)
        else:
            weather_future = None
        
        # === Future covariate fusion ===
        Z_future_reshaped = Z_future.reshape(B * S, H, self.hidden_dim)
        
        if use_future_date and date_future is not None:
            date_future_exp = date_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_date_fusion(Z_future_reshaped, date_future_exp)
            
        if use_future_event and event_future is not None:
            event_future_exp = event_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_event_fusion(Z_future_reshaped, event_future_exp)
            
        if use_future_weather and weather_future is not None:
            weather_future_exp = weather_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_weather_fusion(Z_future_reshaped, weather_future_exp)
        
        Z_future = Z_future_reshaped.view(B, S, H, self.hidden_dim)
        
        # === Prediction ===
        predictions = self.pred_head(Z_future).squeeze(-1)
        predictions = predictions.transpose(1, 2)
        predictions = predictions + fm_preds
        
        return predictions
    
"""  
    
    # MODULAR ARCHITECTURE WITH CONFIGURABLE ORDER
# ============================================================================
# Add this to your SpatialTemporalFM class

class SpatialTemporalFM(nn.Module):
    def __init__(
        self,
        fm_encoder,
        num_sensors,
        input_len,
        output_len,
        date_dim=3,
        event_dim=0,
        weather_dim=5,
        fm_hidden_dim=6144,
        hidden_dim=256,
        graph_layers=2,
        use_road_network=False,
        road_adj=None,
        module_order="past_graph_future",
        # Covariate flags — determine which modules are built (exact param count)
        use_date=True, use_event=True, use_weather=True,
        use_past_date=None, use_past_event=None, use_past_weather=None,
        use_future_date=None, use_future_event=None, use_future_weather=None,
        raw_fm=False,
        device=None
    ):

        super().__init__()
        self.device = device
        self.fm_encoder = fm_encoder
        self.num_sensors = num_sensors
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.fm_hidden_dim = fm_hidden_dim
        self.module_order = module_order
        self.raw_fm = raw_fm

        # Device reference buffer — always present so .to(device) and device queries work
        # even when the model has zero trainable parameters (raw_fm mode).
        self.register_buffer('_device_ref', torch.zeros(1))

        # raw_fm: pure FM output, no adaptation — zero trainable parameters
        if raw_fm:
            return

        # FM wrappers freeze themselves internally; no action needed here.

        # Resolve specific past/future flags (same logic as forward())
        _past_date    = use_date    if use_past_date    is None else use_past_date
        _past_event   = use_event   if use_past_event   is None else use_past_event
        _past_weather = use_weather if use_past_weather is None else use_past_weather
        _fut_date     = use_date    if use_future_date  is None else use_future_date
        _fut_event    = use_event   if use_future_event is None else use_future_event
        _fut_weather  = use_weather if use_future_weather is None else use_future_weather

        # graph_only uses no covariates regardless of flags
        if module_order == "graph_only":
            _past_date = _past_event = _past_weather = False
            _fut_date  = _fut_event  = _fut_weather  = False

        _needs_date    = _past_date    or _fut_date
        _needs_event   = _past_event   or _fut_event
        _needs_weather = _past_weather or _fut_weather
        _needs_any_cov = _needs_date or _needs_event or _needs_weather

        # Covariate encoders — only built when needed
        if _needs_date:
            self.date_encoder = DateEncoder(date_dim, d_model=128)
        if _needs_event:
            self.event_encoder = EventEncoder(event_dim, d_model=128)
        if _needs_weather:
            self.weather_encoder = WeatherEncoder(weather_dim, d_model=128)
        if _needs_any_cov:
            self.covariate_proj = nn.Linear(128, hidden_dim)

        # FM adapter: (B, D_fm, S) -> (B, S, hidden_dim)
        self.fm_adapter = nn.Sequential(
            nn.Linear(fm_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Positional encoding
        self.spatial_pos = nn.Parameter(torch.randn(1, num_sensors, 1, hidden_dim))

        # Past covariate fusion — only built when that covariate is used in the past
        if _past_date:
            self.past_date_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        if _past_event:
            self.past_event_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        if _past_weather:
            self.past_weather_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)

        # Spatial graph layers — skipped for cov_only (pure covariate ablation, no spatial)
        if module_order != "cov_only":
            self.graph_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, num_heads=8) for _ in range(graph_layers)
            ])
            self.use_road_network = use_road_network
            if use_road_network and road_adj is not None:
                self.register_buffer('road_adj', road_adj)
            else:
                self.road_adj = None
        else:
            self.use_road_network = False
            self.road_adj = None

        # Future projection (always needed: maps single step → H steps)
        self.future_proj = nn.Linear(hidden_dim, output_len * hidden_dim)

        # Future covariate fusion — only built when that covariate is used in the future
        if _fut_date:
            self.future_date_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        if _fut_event:
            self.future_event_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        if _fut_weather:
            self.future_weather_fusion = CrossAttentionFusion(hidden_dim, num_heads=8)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    
    def _apply_past_covariates(self, Z, date_X, event_X, weather_X,
                                use_past_date, use_past_event, use_past_weather):
        """Apply past covariate cross-attention (only touches modules that exist)"""
        B, S, _, D = Z.shape
        T = date_X.shape[1]

        # Encode — only call encoder if flag is set AND module was built
        date_past    = self.date_encoder(date_X)    if (use_past_date    and hasattr(self, 'date_encoder'))                       else None
        event_past   = self.event_encoder(event_X)  if (use_past_event   and event_X is not None and hasattr(self, 'event_encoder')) else None
        weather_past = self.weather_encoder(weather_X) if (use_past_weather and hasattr(self, 'weather_encoder'))                  else None

        if date_past    is not None: date_past    = self.covariate_proj(date_past.to(self.device))
        if event_past   is not None: event_past   = self.covariate_proj(event_past.to(self.device))
        if weather_past is not None: weather_past = self.covariate_proj(weather_past.to(self.device))

        # Cross-attention fusion
        Z_query = Z.squeeze(2).unsqueeze(1).reshape(B * S, 1, D)

        if date_past is not None and hasattr(self, 'past_date_fusion'):
            date_past_exp = date_past.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)
            Z_query = self.past_date_fusion(Z_query, date_past_exp)

        if event_past is not None and hasattr(self, 'past_event_fusion'):
            event_past_exp = event_past.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)
            Z_query = self.past_event_fusion(Z_query, event_past_exp)

        if weather_past is not None and hasattr(self, 'past_weather_fusion'):
            weather_past_exp = weather_past.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)
            Z_query = self.past_weather_fusion(Z_query, weather_past_exp)

        return Z_query.view(B, S, 1, D)
    
    def _apply_graph_layers(self, Z):
        """Apply spatial graph attention layers (no-op when graph_layers not built)"""
        if not hasattr(self, 'graph_layers'):
            return Z
        for graph_layer in self.graph_layers:
            Z = graph_layer(Z, adj=self.road_adj)
        return Z
    
    def _apply_future_covariates(self, Z_future, date_Y, event_Y, weather_Y,
                                  use_future_date, use_future_event, use_future_weather):
        """Apply future covariate cross-attention (only touches modules that exist)"""
        B, S, H, D = Z_future.shape

        # Encode — only call encoder if flag is set AND module was built
        date_future    = self.date_encoder(date_Y)    if (use_future_date    and hasattr(self, 'date_encoder'))                        else None
        event_future   = self.event_encoder(event_Y)  if (use_future_event   and event_Y is not None and hasattr(self, 'event_encoder')) else None
        weather_future = self.weather_encoder(weather_Y) if (use_future_weather and hasattr(self, 'weather_encoder'))                   else None

        if date_future    is not None: date_future    = self.covariate_proj(date_future.to(self.device))
        if event_future   is not None: event_future   = self.covariate_proj(event_future.to(self.device))
        if weather_future is not None: weather_future = self.covariate_proj(weather_future.to(self.device))

        # Cross-attention fusion
        Z_future_reshaped = Z_future.reshape(B * S, H, D)

        if date_future is not None and hasattr(self, 'future_date_fusion'):
            date_future_exp = date_future.unsqueeze(1).expand(B, S, H, D).reshape(B * S, H, D)
            Z_future_reshaped = self.future_date_fusion(Z_future_reshaped, date_future_exp)

        if event_future is not None and hasattr(self, 'future_event_fusion'):
            event_future_exp = event_future.unsqueeze(1).expand(B, S, H, D).reshape(B * S, H, D)
            Z_future_reshaped = self.future_event_fusion(Z_future_reshaped, event_future_exp)

        if weather_future is not None and hasattr(self, 'future_weather_fusion'):
            weather_future_exp = weather_future.unsqueeze(1).expand(B, S, H, D).reshape(B * S, H, D)
            Z_future_reshaped = self.future_weather_fusion(Z_future_reshaped, weather_future_exp)

        return Z_future_reshaped.view(B, S, H, D)
    
    def forward(self, x, date_X, event_X, weather_X, date_Y, event_Y, weather_Y, 
                use_date=True, use_event=True, use_weather=True, 
                use_past_date=None, use_past_event=None, use_past_weather=None,
                use_future_date=None, use_future_event=None, use_future_weather=None,
                raw_FM=False):
        
        B, T, S = x.shape
        H = self.output_len
        
        # Resolve flags
        use_past_date = use_date if use_past_date is None else use_past_date
        use_past_event = use_event if use_past_event is None else use_past_event
        use_past_weather = use_weather if use_past_weather is None else use_past_weather
        use_future_date = use_date if use_future_date is None else use_future_date
        use_future_event = use_event if use_future_event is None else use_future_event
        use_future_weather = use_weather if use_future_weather is None else use_future_weather
        
        # FM predictions
        with torch.no_grad():
            fm_preds = self.fm_encoder.predict(x, H)

        if raw_FM or self.raw_fm:
            return fm_preds

        # FM hidden states (only needed for adapter path)
        with torch.no_grad():
            fm_hidden = self.fm_encoder.encode_hidden(x, pool='mean')
        
        # Initial projection
        fm_hidden = fm_hidden.transpose(1, 2)
        Z = self.fm_adapter(fm_hidden)
        Z = Z.unsqueeze(2) + self.spatial_pos  # (B, S, 1, hidden_dim)
        
        # ====================================================================
        # APPLY MODULES IN CONFIGURED ORDER
        # ====================================================================
        
        if self.module_order == "past_graph_future":
            # Current order: Past Cov → Graph → Future Cov
            Z = self._apply_past_covariates(Z, date_X, event_X, weather_X,
                                        use_past_date, use_past_event, use_past_weather)
            Z = self._apply_graph_layers(Z)
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
            Z_future = self._apply_future_covariates(Z_future, date_Y, event_Y, weather_Y,
                                                    use_future_date, use_future_event, use_future_weather)
        
        elif self.module_order == "graph_past_future":
            # Graph → Past Cov → Future Cov
            Z = self._apply_graph_layers(Z)
            Z = self._apply_past_covariates(Z, date_X, event_X, weather_X,
                                        use_past_date, use_past_event, use_past_weather)
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
            Z_future = self._apply_future_covariates(Z_future, date_Y, event_Y, weather_Y,
                                                    use_future_date, use_future_event, use_future_weather)
        
        elif self.module_order == "past_future_graph":
            # Past Cov → Future Cov → Graph
            # Note: Need to handle future projection differently here
            Z = self._apply_past_covariates(Z, date_X, event_X, weather_X,
                                        use_past_date, use_past_event, use_past_weather)
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
            Z_future = self._apply_future_covariates(Z_future, date_Y, event_Y, weather_Y,
                                                    use_future_date, use_future_event, use_future_weather)
            # Apply graph on future representation
            Z_future = self._apply_graph_layers(Z_future)
        
        elif self.module_order == "cov_only":
            # Covariates only: Past Cov → Future Cov (no spatial graph)
            Z = self._apply_past_covariates(Z, date_X, event_X, weather_X,
                                        use_past_date, use_past_event, use_past_weather)
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
            Z_future = self._apply_future_covariates(Z_future, date_Y, event_Y, weather_Y,
                                                    use_future_date, use_future_event, use_future_weather)

        elif self.module_order == "graph_only":
            # Graph only (no covariates for comparison)
            Z = self._apply_graph_layers(Z)
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
        
        elif self.module_order == "future_graph_past":
            # Reversed: Future Cov → Graph → Past Cov
            Z_spatial = Z.squeeze(2)
            Z_future = self.future_proj(Z_spatial).view(B, S, H, self.hidden_dim)
            Z_future = self._apply_future_covariates(Z_future, date_Y, event_Y, weather_Y,
                                                    use_future_date, use_future_event, use_future_weather)
            Z_future = self._apply_graph_layers(Z_future)
            # Past covariates applied differently here - expand to H dimension
            # (This is experimental - may need different fusion strategy)
            Z = Z.expand(B, S, H, self.hidden_dim)
            Z = self._apply_past_covariates(Z[:, :, 0:1, :], date_X, event_X, weather_X,
                                        use_past_date, use_past_event, use_past_weather)
            Z_future = Z_future + Z.expand(B, S, H, self.hidden_dim)
        
        else:
            raise ValueError(f"Unknown module_order: {self.module_order}")
        
        # Final prediction
        predictions = self.pred_head(Z_future).squeeze(-1)
        predictions = predictions.transpose(1, 2)
        predictions = predictions + fm_preds
        
        return predictions
    
# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, epochs=50, lr=3e-4, model_name="model",
                use_date=True, use_event=True, use_weather=True,
                use_past_date=None, use_past_event=None, use_past_weather=None,
                use_future_date=None, use_future_event=None, use_future_weather=None, raw_fm=False):
    """Train only adapter modules with conditional covariate usage"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    device = model._device_ref.device

    epoch_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            x = batch["x"].to(device)
            y_true = batch["y"].to(device)
            y_mask = batch.get("y_mask", None)
            if y_mask is not None:
                y_mask = y_mask.to(device)
            
            # Load covariates with correct batch keys
            date_X = batch["x_calendar"].to(device)  # (B, T, K_date)
            date_Y = batch["y_calendar"].to(device)  # (B, H, K_date)
            event_X = batch["x_events"].to(device) if batch.get("x_events") is not None else None
            event_Y = batch["y_events"].to(device) if batch.get("y_events") is not None else None
            weather_X = batch["x_weather"].to(device)  # (B, T, K_weather)
            weather_Y = batch["y_weather"].to(device)  # (B, H, K_weather)

            # Forward with conditional covariate usage
            y_pred = model(x, date_X, event_X, weather_X, date_Y, event_Y, weather_Y,
                            use_date=use_date, use_event=use_event, use_weather=use_weather,
                            use_past_date=use_past_date, use_past_event=use_past_event, use_past_weather=use_past_weather,
                            use_future_date=use_future_date, use_future_event=use_future_event, use_future_weather=use_future_weather)
            # Loss
            if y_mask is not None:
                loss = (y_mask * (y_pred - y_true).abs()).sum() / (y_mask.sum() + 1e-6)
            else:
                loss = nn.L1Loss()(y_pred, y_true)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"\rEpoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}", end="", flush=True)
    
    print()  # New line after training
    model.loss_history = epoch_losses
    train_time = time.time() - start_time
    model.train_time = train_time
    
    # Plot
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MAE)")
    plt.title(f"Spatial-Temporal FM Training")
    plt.savefig(f"results_server/loss_functions/loss_{model_name}.png")
    plt.close()
    
    return train_time

@torch.no_grad()
def evaluate_model(model, test_loader, norm_params, 
                    use_date=True, use_event=True, use_weather=True,
                    use_past_date=None, use_past_event=None, use_past_weather=None,
                    use_future_date=None, use_future_event=None, use_future_weather=None,
                    raw_fm=False):
    """Evaluate and denormalize with conditional covariate usage"""
    model.eval()
    device = model._device_ref.device
    
    y_true_list, y_pred_list = [], []
    start_time = time.time()
    
    for batch in test_loader:
        x = batch["x"].to(device)
        y_true = batch["y"].to(device)
        
        date_X = batch["x_calendar"].to(device)  # (B, T, K_date)
        date_Y = batch["y_calendar"].to(device)  # (B, H, K_date)
        event_X = batch["x_events"].to(device) if batch.get("x_events") is not None else None
        event_Y = batch["y_events"].to(device) if batch.get("y_events") is not None else None
        weather_X = batch["x_weather"].to(device)  # (B, T, K_weather)
        weather_Y = batch["y_weather"].to(device)  # (B, H, K_weather)

        y_pred = model(x, date_X, event_X, weather_X, date_Y, event_Y, weather_Y,
                        use_date=use_date, use_event=use_event, use_weather=use_weather,
                        use_past_date=use_past_date, use_past_event=use_past_event, use_past_weather=use_past_weather,
                        use_future_date=use_future_date, use_future_event=use_future_event, use_future_weather=use_future_weather,
                        raw_FM=raw_fm)
        
        y_true_list.append(y_true.cpu())
        y_pred_list.append(y_pred.cpu())
    
    model.eval_time = time.time() - start_time
    
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    
    # Denormalize
    diff = norm_params['X_max'] - norm_params['X_min']
    y_min = norm_params['X_min']
    y_true_denorm = y_true * diff + y_min
    y_pred_denorm = y_pred * diff + y_min
    
    return y_true_denorm.numpy(), y_pred_denorm.numpy()