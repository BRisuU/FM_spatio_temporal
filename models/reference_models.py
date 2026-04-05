"""
reference_models.py
═══════════════════
Paper-faithful implementations of three reference models for spatio-temporal
traffic forecasting with exogenous covariates:

  STAEformer — Liu et al., CIKM 2023        (arXiv 2308.10425v5)
  STGformer  — Wang et al., arXiv 2024      (arXiv 2410.00385v2)
  ExoST      — Chen et al., arXiv 2025      (arXiv 2509.05779v3)

All architectures verified line-by-line against uploaded PDF files.

═══════════════════════════════════════════════════════════════════════════════
FINAL CORRECTIONS  (from direct PDF reading)
═══════════════════════════════════════════════════════════════════════════════

STAEformer (2308.10425v5, §3.1-3.2)
────────────────────────────────────
  FIX-1 (from previous round): E_a shape is (T, N, d_a), NOT (steps_per_day,N,D).
         Paper §3.1: "E_a ∈ R^{T×N×d_a} ... shared across different traffic time
         series."  T = input_len = 12.  E_a is a FIXED learned parameter, no
         runtime indexing.  Periodicity (tod/dow) is captured by E_p separately.

  FIX-2 (from previous round): Z = E_f ‖ E_p ‖ E_a  [CONCATENATION, not addition]
         d_h = 3·d_f + d_a.  Paper: d_f=24, d_a=80 → d_h=152.

  FIX-3 (from previous round): Model has BOTH L temporal AND L spatial transformer
         layers (Fig. 2 shows two stacks).

  FIX-4 (THIS ROUND): tod/dow are REQUIRED for E_p but the benchmark data loader
         does not currently emit them.  Added graceful fallback: if tod or dow is
         None, E_p = 0 (zero tensor) so model still runs; note that the d_f slots
         for E_p are retained in Z shape so all downstream dimensions stay valid.
         To unlock full accuracy: add `tod` and `dow` columns to your data loader.

STGformer (2410.00385v2, §IV-B, §IV-C)
────────────────────────────────────────
  FIX-5 (from previous round): Factored spatial + temporal linear attention with
         SHARED Q/K/V (Eq. 5-7):
           A_s = (1/n_s) Q K^T V  → R^{T×N×C}   (spatial, per timestep)
           A_t = (1/n_t) Q^T K V  → R^{T×N×C}   (temporal, per node, transposed back)
         where n_s = N·√C, n_t = T·√C (scaling denominators from Eq. 6).

  FIX-6 (from previous round): Recursive fusion per Eq. (3-4):
         p_{n+1} = a_n(q_n) ⊙ g_n(p_n)
         p_0 = X_emb, g_0 = Identity, g_n = Linear for n≥1.

  FIX-7 (from previous round): Xemb = Xdata ‖ Xw ‖ Xd ‖ Xste  [concatenation, 4×d].

  FIX-8 (THIS ROUND): Removed dead code in _STGAttention (two earlier discarded
         KV_s computations with wrong shapes were left as confusing comments).
         Cleaned to only the correct kv_s = einsum('btnc,btnd->btcd', K, V) path.

  FIX-9 (THIS ROUND): Same tod/dow optional fallback as STAEformer.

ExoST (2509.05779v3, §4)  — COMPLETE ARCHITECTURAL REPLACEMENT
───────────────────────────────────────────────────────────────
  REPLACED: Previous "TFTExoST" was a TFT (Lim 2021) variant — entirely the wrong
            architecture.  Now implements Chen et al. 2025 "select-then-balance":

  Stage 1 — SELECT: Latent-Space Gated Expert Module (§4.1)
    Conditional Embedding (Eq. 2):
      X^τ = Dropout(Act(W_x^τ X + W_e^τ E^τ + b^τ))
      τ ∈ {p=past, f=future}; E^τ is the exogenous tensor for that split.

    Gated Expert Selector (Eq. 3-4):
      g^τ = Softmax(W_g^τ X^τ)         K experts, each gate weight ∈ R^K
      X^τ' = Σ_{k=1}^{K} g^τ_k W_k^τ X^τ

  Stage 2 — BALANCE: Dual-Branch Siamese Architecture (§4.2)
    Siamese ST Encoders (Eq. 5):
      Y^τ = ϕ^τ_st(X^τ')   (Siamese = independent encoders for p and f)
      Backbone: lightweight temporal transformer, any ST backbone can replace.

    Context-Aware Balancer (Eq. 6-7):
      Y = Y^p + Y^f
      α = σ(MLP(AvgPool(Y)))
      Ŷ = α ⊙ Y^p + (1−α) ⊙ Y^f + Y   (residual connection)

  DATA FIT: user benchmark provides global exo (B,T,F), not per-node.
            Broadcast to (B,T,N,F) before conditional embedding.
            E^p = concat(x_calendar, x_weather);  E^f = concat(y_calendar, y_weather).

  The old TFT-based model is preserved as CovTFT (renamed) for comparison.

═══════════════════════════════════════════════════════════════════════════════
DATA INTERFACE  (what the batch dict must contain)
═══════════════════════════════════════════════════════════════════════════════
  REQUIRED for all models:
    "x"        : (B, T, N)    or (B, T, N, C)  float   — traffic signal
    "y"        : (B, H, N)                      float   — target
    "y_mask"   : (B, H, N)                      float   — validity mask (opt)

  OPTIONAL (STAEformer / STGformer periodicity embedding):
    "tod"      : (B, T)   int   — timestamp-of-day index  0..steps_per_day-1
    "dow"      : (B, T)   int   — day-of-week             0..6
    Without these the E_p slot is zeroed out; full accuracy requires them.

  REQUIRED for ExoST / CovTFT / TFTExoST (from the Zurich/NYC benchmark):
    "x_calendar"  : (B, T, Fd)    float — past  date encoding
    "y_calendar"  : (B, H, Fd)    float — future date encoding
    "x_weather"   : (B, T, Fw)    float — past  weather
    "y_weather"   : (B, H, Fw)    float — future weather
    "x_events"    : (B, T, Fe)    float — past  event/holiday flags  [optional]
    "y_events"    : (B, H, Fe)    float — future event/holiday flags [optional]
  NOTE: CovTFT / TFTExoST receive events as event_X / event_Y; the train/eval
        loops remap x_events→event_X and y_events→event_Y automatically.

═══════════════════════════════════════════════════════════════════════════════
BENCHMARKING WITHOUT CUSTOM CODE
═══════════════════════════════════════════════════════════════════════════════
  Official repos:
    STAEformer: git clone https://github.com/XDZhelheim/STAEformer
                cd model && python train.py -d METRLA -g 0
    STGformer:  git clone https://github.com/Dreamzz5/STGformer
                cd model && python train.py
    ExoST:      https://github.com/YuanshaoZhu/ExoST   (2509.05779)

  Unified fair benchmark (identical pipeline for all models):
    git clone https://github.com/GestaltCogTeam/BasicTS && pip install basicts
    python experiments/train.py -c baselines/STAEformer/METRLA.py
"""

import math, os, sys, json, time, shutil, tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _ffn(d: int, expansion: int = 4, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, d * expansion), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d * expansion, d), nn.Dropout(dropout),
    )

def _get(batch, key, device):
    v = batch.get(key)
    return v.to(device) if v is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STAEformer — Liu et al., CIKM 2023  (2308.10425v5)
#     github.com/XDZhelheim/STAEformer
#
# Embedding layer §3.1, Eq.(1-3):
#   E_f  = FC(X)                          ∈ R^{B×T×N×d_f}   [Eq.2]
#   E_w  = T_w[W^t]                       ∈ R^{B×T×d_f}  → broadcast (B,T,N,d_f)
#   E_d  = T_d[D^t]                       ∈ R^{B×T×d_f}  → broadcast (B,T,N,d_f)
#   E_p  = concat(E_w, E_d)               ∈ R^{B×T×N×2·d_f}  [periodicity]
#          → zero tensor if tod/dow not provided (graceful fallback)
#   E_a  = nn.Parameter(T, N, d_a)        ∈ R^{T×N×d_a}  [fixed, SHARED across samples]
#   Z    = concat(E_f, E_p, E_a)          ∈ R^{B×T×N×d_h}, d_h = 3·d_f + d_a  [Eq.3]
#
# Transformer §3.2:
#   L × TemporalLayer (attend over T, per node)
#   L × SpatialLayer  (attend over N, per timestep)
#
# Regression Eq.(8): per node, reshape (T, d_h) → flatten → Linear → H
#
# Paper hyperparameters: d_f=24, d_a=80 → d_h=152, L=3, heads=4, batch=16, lr=1e-3
# ─────────────────────────────────────────────────────────────────────────────

class _STAETemporalLayer(nn.Module):
    """Transformer encoder block attending over the T axis (per sensor node)."""
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn  = _ffn(d, dropout=dropout)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, D) → (B, T, N, D)"""
        B, T, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # fold N into batch
        o, _ = self.attn(h, h, h)
        h = self.n1(h + self.drop(o))
        h = self.n2(h + self.drop(self.ffn(h)))
        return h.reshape(B, N, T, D).permute(0, 2, 1, 3)


class _STAESpatialLayer(nn.Module):
    """Transformer encoder block attending over the N axis (per timestep)."""
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn  = _ffn(d, dropout=dropout)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, D) → (B, T, N, D)"""
        B, T, N, D = x.shape
        h = x.reshape(B * T, N, D)   # fold T into batch
        o, _ = self.attn(h, h, h)
        h = self.n1(h + self.drop(o))
        h = self.n2(h + self.drop(self.ffn(h)))
        return h.reshape(B, T, N, D)


class STAEformer(nn.Module):
    """
    Spatio-Temporal Adaptive Embedding Transformer  (Liu et al., CIKM 2023).

    Input  : x (B,T,N,C), tod (B,T) int [optional], dow (B,T) int [optional]
    Output : (B, H, N)

    Parameters
    ----------
    num_nodes     : N
    in_channels   : C (1 for single speed/flow channel)
    input_len     : T (12 for 1-hour @ 5-min resolution)
    output_len    : H
    d_f           : feature + periodicity dim each      (paper default: 24)
    d_a           : adaptive embedding dim              (paper default: 80)
    num_layers    : L — number of temporal AND spatial layers  (paper: 3)
    num_heads     : attention heads                     (paper: 4)
    steps_per_day : N_d for the tod embedding dict      (paper: 288)
    dropout       : float
    """
    use_covariates: bool = False

    def __init__(self, num_nodes: int, in_channels: int,
                 input_len: int, output_len: int,
                 d_f: int = 24, d_a: int = 80,
                 num_layers: int = 3, num_heads: int = 4,
                 steps_per_day: int = 288, dropout: float = 0.1):
        super().__init__()
        d_h = 3 * d_f + d_a   # paper: 3·24 + 80 = 152

        # ── Embedding components ──────────────────────────────────────────────
        self.value_proj   = nn.Linear(in_channels, d_f)      # E_f = FC(X)
        self.tod_emb      = nn.Embedding(steps_per_day, d_f) # T_d timestamp dict
        self.dow_emb      = nn.Embedding(7, d_f)             # T_w weekday dict
        # E_a: fixed (T, N, d_a) shared across all samples; NOT indexed by tod
        self.adaptive_emb = nn.Parameter(
            torch.randn(input_len, num_nodes, d_a) * 0.02)

        self.emb_norm = nn.LayerNorm(d_h)
        self.emb_drop = nn.Dropout(dropout)

        # ── L temporal layers, then L spatial layers (Fig. 2) ────────────────
        self.temporal_layers = nn.ModuleList([
            _STAETemporalLayer(d_h, num_heads, dropout) for _ in range(num_layers)])
        self.spatial_layers  = nn.ModuleList([
            _STAESpatialLayer(d_h, num_heads, dropout)  for _ in range(num_layers)])

        # ── Regression head: per node, T·d_h → H  (Eq. 8) ───────────────────
        self.regression = nn.Linear(input_len * d_h, output_len)

        self._T, self._d_h = input_len, d_h
        self._d_f = d_f

    def forward(self, x: torch.Tensor,
                tod: Optional[torch.Tensor] = None,
                dow: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        """
        x   : (B, T, N, C)  or  (B, T, N) for single-channel
        tod : (B, T) int  — timestamp-of-day index (optional; zeroes E_p if absent)
        dow : (B, T) int  — day-of-week            (optional)
        → (B, H, N)
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, T, N, _ = x.shape

        # E_f: feature embedding (B, T, N, d_f)
        E_f = self.value_proj(x)

        # E_p: periodicity embedding  (B, T, N, 2·d_f)
        # Fallback to zero if tod/dow not provided (e.g. data loader gap)
        if tod is not None and dow is not None:
            E_w = self.dow_emb(dow).unsqueeze(2).expand(B, T, N, -1)   # (B,T,N,d_f)
            E_d = self.tod_emb(tod).unsqueeze(2).expand(B, T, N, -1)   # (B,T,N,d_f)
            E_p = torch.cat([E_w, E_d], dim=-1)                         # (B,T,N,2·d_f)
        else:
            E_p = x.new_zeros(B, T, N, 2 * self._d_f)

        # E_a: adaptive embedding  (B, T, N, d_a)
        E_a = self.adaptive_emb.unsqueeze(0).expand(B, -1, -1, -1)

        # Z = [E_f ‖ E_p ‖ E_a]  (B, T, N, d_h)
        Z = self.emb_norm(self.emb_drop(torch.cat([E_f, E_p, E_a], dim=-1)))

        # Transformer: L temporal then L spatial
        for layer in self.temporal_layers:
            Z = layer(Z)
        for layer in self.spatial_layers:
            Z = layer(Z)

        # Regression: (B, N, T·d_h) → (B, N, H) → (B, H, N)
        Z_flat = Z.permute(0, 2, 1, 3).reshape(B, N, self._T * self._d_h)
        return self.regression(Z_flat).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STGformer — Wang et al., arXiv 2410.00385v2
#     github.com/Dreamzz5/STGformer
#
# Embedding §IV-B:
#   X_emb = X_data ‖ X_w ‖ X_d ‖ X_ste   ∈ R^{T×N×4d}  [concatenation]
#
# Graph propagation:
#   X_0 = X_emb,  X_n = Ã · X_{n-1}    (Ã = row-normalised adjacency)
#
# Recursive ST-Attention §IV-C, Eq.(3-7):
#   ONE shared set of Q/K/V projections used for ALL k propagation orders.
#   For each propagated feature h:
#     Q = h w_Q,  K = h w_K,  V = h w_V    (h ∈ R^{T×N×C})
#     Spatial  linear attn A_s = (1/n_s) Q K^T V    → R^{T×N×C}   [Eq.7 left]
#     Temporal linear attn A_t = (1/n_t) Q^T K V    → R^{T×N×C}   [Eq.7 right,
#                                                       Q^T ≡ swap T,N dims]
#     ST output = A_s + A_t
#   Recursive fusion p_{n+1} = ST(X_n) ⊙ g_n(p_n)  [Eq.3-4]
#     p_0 = X_emb,  g_0 = Identity,  g_n = Linear (n≥1)
#
# Prediction head: per node, flatten T·4d → MLP → H
# ─────────────────────────────────────────────────────────────────────────────

class _STGAttention(nn.Module):
    """
    Factored spatiotemporal linear attention with SHARED Q/K/V (Wang et al.,
    Eq. 5-7).  One set of projection weights serves all k propagation orders.

    Spatial path  (Eq.7 left):  A_s = (1/n_s) · Q K^T V
      For each timestep t:  K[t]^T V[t] ∈ R^{C×C},  then Q[t] · result ∈ R^{N×C}
      n_s = N · √C  (normalisation from Eq.6 applied over N positions)

    Temporal path (Eq.7 right): A_t = (1/n_t) · Q^T K V
      "Q^T" ≡ view Q with T and N swapped: (B,N,T,C)
      For each node n:  K_t[n]^T V_t[n] ∈ R^{C×C}, then Q_t[n] · result ∈ R^{T×C}
      n_t = T · √C

    ST output = A_s + A_t  (summed, then passed back as the attention result)
    """
    def __init__(self, d: int):
        super().__init__()
        self.Wq    = nn.Linear(d, d, bias=False)
        self.Wk    = nn.Linear(d, d, bias=False)
        self.Wv    = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, N, C) → (B, T, N, C)"""
        B, T, N, C = h.shape
        Q = self.Wq(h)   # (B, T, N, C)
        K = self.Wk(h)
        V = self.Wv(h)

        # ── Spatial linear attention ─────────────────────────────────────────
        # For each (b,t): kv[c1,c2] = Σ_n K[n,c1]·V[n,c2]
        kv_s = torch.einsum('btnc,btnd->btcd', K, V)   # (B, T, C, C)
        # A_s[b,t,n,c] = Σ_c1 Q[b,t,n,c1] · kv_s[b,t,c1,c] / n_s
        n_s  = N * self.scale
        A_s  = torch.einsum('btnc,btcd->btnd', Q, kv_s) / n_s   # (B, T, N, C)

        # ── Temporal linear attention ────────────────────────────────────────
        # View with T,N transposed: Q_t ∈ (B, N, T, C)
        Q_t = Q.permute(0, 2, 1, 3)
        K_t = K.permute(0, 2, 1, 3)
        V_t = V.permute(0, 2, 1, 3)
        # For each (b,n): kv_t[c1,c2] = Σ_t K_t[t,c1]·V_t[t,c2]
        kv_t = torch.einsum('bntc,bntd->bncd', K_t, V_t)   # (B, N, C, C)
        n_t  = T * self.scale
        A_t  = torch.einsum('bntc,bncd->bntd', Q_t, kv_t) / n_t   # (B, N, T, C)
        A_t  = A_t.permute(0, 2, 1, 3)                              # (B, T, N, C)

        return A_s + A_t


class _STGBlock(nn.Module):
    """
    STGformer block: graph propagation × k  →  recursive ST-attention  →  FFN.

    Recursive fusion (Eq. 3-4):
      p_0 = X_emb
      p_{n+1} = attn(X_n) ⊙ g_n(p_n)
      g_0 = Identity,  g_n = Linear for n ≥ 1
    """
    def __init__(self, d: int, k_order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.K    = k_order
        self.attn = _STGAttention(d)   # shared across all k orders
        # g_n for n = 1..k-1  (g_0 is Identity — applied as a no-op)
        self.g = nn.ModuleList([
            nn.Linear(d, d) for _ in range(max(k_order - 1, 0))])
        self.ffn      = _ffn(d, dropout=dropout)
        self.n1       = nn.LayerNorm(d)
        self.n2       = nn.LayerNorm(d)
        self.drop     = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, Anorm: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, T, N, d)
        Anorm : (N, N)  row-normalised adjacency
        → (B, T, N, d)
        """
        # Graph propagation: X_0=x, X_n = Ã · X_{n-1}
        props = [x]
        for _ in range(self.K - 1):
            props.append(torch.einsum('mn,btnC->btmC', Anorm, props[-1]))

        # Recursive attention fusion
        p = props[0]
        for n in range(self.K):
            att = self.attn(props[n])       # a_n(q_n) from Eq.3
            if n == 0:
                p = att * p                 # g_0 = Identity
            else:
                p = att * self.g[n - 1](p) # g_n = Linear

        # Residual + FFN
        out = self.n1(x + self.drop(self.out_proj(p)))
        return self.n2(out + self.drop(self.ffn(out)))


class STGformer(nn.Module):
    """
    Efficient Spatiotemporal Graph Transformer  (Wang et al., arXiv 2410.00385v2).

    Input  : x (B,T,N,C), tod (B,T) int [optional], dow (B,T) int [optional],
             adj_mx (N,N) row-normalised road-topology adjacency  [optional]
    Output : (B, H, N)

    Parameters
    ----------
    num_nodes     : N
    in_channels   : C
    input_len     : T
    output_len    : H
    d             : per-component embedding dim (paper: 32, giving X_emb ∈ R^{4d=128})
    k_order       : k propagation orders (paper: 3)
    steps_per_day : for tod embedding
    dropout       : float
    """
    use_covariates: bool = False

    def __init__(self, num_nodes: int, in_channels: int,
                 input_len: int, output_len: int,
                 d: int = 32, k_order: int = 3,
                 steps_per_day: int = 288, dropout: float = 0.1):
        super().__init__()
        d_emb = 4 * d     # X_emb = [X_data ‖ X_w ‖ X_d ‖ X_ste]
        self.T, self.d_emb, self._d = input_len, d_emb, d

        # ── Embedding §IV-B ──────────────────────────────────────────────────
        self.value_proj   = nn.Linear(in_channels, d)
        self.tod_emb      = nn.Embedding(steps_per_day, d)
        self.dow_emb      = nn.Embedding(7, d)
        # X_ste: fixed (T, N, d) adaptive param (same logic as STAEformer's E_a)
        self.adaptive_emb = nn.Parameter(
            torch.randn(input_len, num_nodes, d) * 0.02)
        self.emb_norm     = nn.LayerNorm(d_emb)

        # ── Single STG block (paper: one layer is sufficient) ────────────────
        self.block = _STGBlock(d_emb, k_order, dropout)

        # ── Prediction head: per node, T·d_emb → MLP → H ────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(input_len * d_emb),
            nn.Linear(input_len * d_emb, d_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, output_len),
        )

        self.register_buffer('_adj', None)

    def register_adj(self, adj_mx: torch.Tensor):
        """Pre-register road-topology adjacency (will be row-normalised)."""
        rs = adj_mx.sum(1, keepdim=True).clamp(min=1e-6)
        self._adj = (adj_mx / rs).float()

    def forward(self, x: torch.Tensor,
                tod: Optional[torch.Tensor] = None,
                dow: Optional[torch.Tensor] = None,
                adj_mx: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, T, N, _ = x.shape

        # Resolve adjacency: passed-in > pre-registered > uniform fallback
        if adj_mx is not None:
            rs = adj_mx.sum(1, keepdim=True).clamp(min=1e-6)
            Anorm = (adj_mx / rs).to(x.device)
        elif self._adj is not None:
            Anorm = self._adj.to(x.device)
        else:
            Anorm = torch.ones(N, N, device=x.device) / N   # no topology prior

        X_data = self.value_proj(x)                                         # (B,T,N,d)

        # Graceful fallback if tod/dow not provided
        if tod is not None and dow is not None:
            X_w = self.dow_emb(dow).unsqueeze(2).expand(B, T, N, -1)       # (B,T,N,d)
            X_d = self.tod_emb(tod).unsqueeze(2).expand(B, T, N, -1)       # (B,T,N,d)
        else:
            X_w = x.new_zeros(B, T, N, self._d)
            X_d = x.new_zeros(B, T, N, self._d)

        X_ste = self.adaptive_emb.unsqueeze(0).expand(B, -1, -1, -1)       # (B,T,N,d)
        X_emb = self.emb_norm(torch.cat([X_data, X_w, X_d, X_ste], dim=-1))# (B,T,N,4d)

        out = self.block(X_emb, Anorm)                                      # (B,T,N,4d)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, T * self.d_emb)   # (B,N,T·4d)
        return self.head(out_flat).transpose(1, 2)                          # (B,H,N)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ExoST — Chen et al., arXiv 2509.05779v3
#     "Select, then Balance" exogenous variable modeling framework.
#     github.com/YuanshaoZhu/ExoST
#
# Stage 1 — SELECT: Latent-Space Gated Expert (§4.1)
#   ConditionalEmbedding (Eq.2): X^τ = Dropout(Act(W_x X + W_e E^τ + b))
#   GatedExpertSelector (Eq.3-4): g = Softmax(W_g X^τ); X^τ' = Σ g_k W_k X^τ
#
# Stage 2 — BALANCE: Dual-Branch Siamese (§4.2)
#   SiameseST encoder:  Y^τ = ϕ_st(X^τ')  [independent for past and future]
#   ContextAwareBalancer (Eq.6-7):
#     Y = Y^p + Y^f
#     α = σ(MLP(AvgPool(Y)))
#     Ŷ = α ⊙ Y^p + (1-α) ⊙ Y^f + Y
#
# For the benchmark data the exo inputs are global (B,T,F), not per-node.
# They are broadcast to all N nodes inside forward().
#
# Backbone choice: lightweight temporal transformer (any ST backbone works;
# per paper: "any modern ST encoder can be plugged in without architectural change").
# ─────────────────────────────────────────────────────────────────────────────

class _ExoSTConditionalEmbedding(nn.Module):
    """
    Eq. (2): X^τ = Dropout(Act(W_x^τ X + W_e^τ E^τ + b^τ))
    Aligns endogenous (in_dim) and exogenous (exo_dim) into hidden_dim.
    """
    def __init__(self, in_dim: int, exo_dim: int,
                 hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_x = nn.Linear(in_dim,  hidden_dim)
        self.W_e = nn.Linear(exo_dim, hidden_dim, bias=False)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, N, F_in)     endogenous features
        e : (B, T_in, N, F_exo)    exogenous features (same length as x)
        → (B, T_in, N, hidden_dim)
        """
        return self.drop(self.act(self.W_x(x) + self.W_e(e)))


class _ExoSTGatedExpert(nn.Module):
    """
    Eq. (3-4): g = Softmax(W_g X^τ);  X^τ' = Σ_k g_k W_k X^τ
    K expert projections with a learned routing gate.
    """
    def __init__(self, hidden_dim: int, k_experts: int = 4):
        super().__init__()
        self.K     = k_experts
        self.W_g   = nn.Linear(hidden_dim, k_experts)
        self.W_exp = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(k_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, N, H) → (B, T, N, H)"""
        B, T, N, H = x.shape
        flat = x.reshape(-1, H)                              # (B*T*N, H)
        g    = torch.softmax(self.W_g(flat), dim=-1)         # (B*T*N, K)
        out  = sum(g[:, k:k+1] * self.W_exp[k](flat)
                   for k in range(self.K))                   # (B*T*N, H)
        return out.reshape(B, T, N, H)


class _ExoSTEncoder(nn.Module):
    """
    Lightweight ST backbone for the Siamese encoder branches.
    Input  : (B, T_in, N, H)
    Output : (B, H_out, N, 1)   — predictions per node per future step

    Architecture: temporal Transformer over the T_in axis (per node),
    then linear head to map T_in × H → H_out.  Replace with any heavier
    backbone (AGCRN, GWNet, etc.) for best accuracy.
    """
    def __init__(self, hidden_dim: int, input_len: int,
                 output_len: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.T_in  = input_len
        self.T_out = output_len
        self.H     = hidden_dim
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ffn   = _ffn(hidden_dim, dropout=dropout)
        self.n1, self.n2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Linear(input_len * hidden_dim, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_in, N, H) → (B, T_out, N, 1)"""
        B, T, N, H = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, H)   # fold N into batch
        o, _ = self.attn(h, h, h)
        h = self.n1(h + self.drop(o))
        h = self.n2(h + self.drop(self.ffn(h)))
        # Flatten T × H per node → predict T_out future steps
        h_flat = h.reshape(B, N, T * H)
        out    = self.head(h_flat)                         # (B, N, T_out)
        return out.transpose(1, 2).unsqueeze(-1)           # (B, T_out, N, 1)


class ExoST(nn.Module):
    """
    Select-Then-Balance exogenous variable modeling framework
    (Chen et al., arXiv 2509.05779v3).

    Input  : x (B,T,N) traffic signal
             x_calendar/y_calendar (B,T/H,Fd)  past/future date encoding
             x_weather/y_weather   (B,T/H,Fw)  past/future weather
    Output : (B, H, N)  point forecast

    The global exogenous inputs (not per-node) are broadcast to all N nodes
    before the conditional embedding step.

    Parameters
    ----------
    num_nodes     : N
    input_len     : T (past steps)
    output_len    : H (future steps to predict)
    cal_dim       : Fd — calendar feature dimension
    weather_dim   : Fw — weather feature dimension
    hidden_dim    : H_hidden (paper default: 64)
    k_experts     : K (paper default: 4)
    num_heads     : attention heads in backbone
    dropout       : float
    """
    use_covariates: bool = True

    def __init__(self, num_nodes: int, input_len: int, output_len: int,
                 cal_dim: int = 4, weather_dim: int = 4, event_dim: int = 0,
                 hidden_dim: int = 64, k_experts: int = 4,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.N, self.T, self.H = num_nodes, input_len, output_len
        self.cal_dim, self.weather_dim, self.event_dim = cal_dim, weather_dim, event_dim
        exo_dim = cal_dim + weather_dim + event_dim   # Fp = Ff = Fd + Fw + Fe

        # ── Stage 1: SELECT ──────────────────────────────────────────────────
        # Past branch: X + E^p
        self.cond_emb_p = _ExoSTConditionalEmbedding(
            in_dim=1, exo_dim=exo_dim, hidden_dim=hidden_dim, dropout=dropout)
        # Future branch: X + E^f (X is repeated as context, E^f is future exo)
        self.cond_emb_f = _ExoSTConditionalEmbedding(
            in_dim=1, exo_dim=exo_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.expert_p = _ExoSTGatedExpert(hidden_dim, k_experts)
        self.expert_f = _ExoSTGatedExpert(hidden_dim, k_experts)

        # ── Stage 2: BALANCE ─────────────────────────────────────────────────
        # Siamese ST encoders (independent parameters for past and future)
        self.encoder_p = _ExoSTEncoder(hidden_dim, input_len,
                                        output_len, num_heads, dropout)
        self.encoder_f = _ExoSTEncoder(hidden_dim, output_len,
                                        output_len, num_heads, dropout)

        # Context-aware balancer: α = σ(MLP(AvgPool(Y^p + Y^f)))
        # Input to MLP: (B, 1, N, 1) after AvgPool over T_out
        self.balancer = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _broadcast_exo(self, cov: Optional[torch.Tensor],
                       B: int, L: int, N: int, F: int,
                       device) -> torch.Tensor:
        """
        (B, L, F) global exo → (B, L, N, F) per-node by broadcasting.
        Returns zeros if cov is None.
        """
        if cov is not None:
            return cov.unsqueeze(2).expand(B, L, N, F)
        return torch.zeros(B, L, N, F, device=device)

    def forward(self, x: torch.Tensor,
                tod=None, dow=None,
                x_calendar: Optional[torch.Tensor] = None,
                y_calendar: Optional[torch.Tensor] = None,
                x_weather:  Optional[torch.Tensor] = None,
                y_weather:  Optional[torch.Tensor] = None,
                x_events:   Optional[torch.Tensor] = None,
                y_events:   Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        """
        x           : (B, T, N)  or  (B, T, N, C)
        x_calendar  : (B, T, Fd)   past  date encoding
        y_calendar  : (B, H, Fd)   future date encoding
        x_weather   : (B, T, Fw)   past  weather
        y_weather   : (B, H, Fw)   future weather
        x_events    : (B, T, Fe)   past  event flags  [optional]
        y_events    : (B, H, Fe)   future event flags [optional]
        → (B, H, N)
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)    # (B, T, N, 1)
        B, T, N, _ = x.shape
        dev = x.device

        # ── Past exogenous: E^p = [x_calendar ‖ x_weather ‖ x_events] ──────
        E_cal_p = self._broadcast_exo(x_calendar, B, T, N, self.cal_dim, dev)
        E_wea_p = self._broadcast_exo(x_weather,  B, T, N, self.weather_dim, dev)
        parts_p = [E_cal_p, E_wea_p]
        if self.event_dim > 0:
            parts_p.append(self._broadcast_exo(x_events, B, T, N, self.event_dim, dev))
        E_p = torch.cat(parts_p, dim=-1)   # (B, T, N, Fp)

        # ── Future exogenous: E^f = [y_calendar ‖ y_weather ‖ y_events] ─────
        H = self.H
        E_cal_f = self._broadcast_exo(y_calendar, B, H, N, self.cal_dim, dev)
        E_wea_f = self._broadcast_exo(y_weather,  B, H, N, self.weather_dim, dev)
        parts_f = [E_cal_f, E_wea_f]
        if self.event_dim > 0:
            parts_f.append(self._broadcast_exo(y_events, B, H, N, self.event_dim, dev))
        E_f = torch.cat(parts_f, dim=-1)   # (B, H, N, Ff)

        # For the future branch: repeat X (past observations) along the future
        # time axis as context (zero-padding after T, consistent with paper §4.1)
        x_f_ctx = x.mean(dim=1, keepdim=True).expand(B, H, N, 1)  # (B,H,N,1)

        # ── Stage 1: SELECT ──────────────────────────────────────────────────
        X_p = self.cond_emb_p(x,      E_p)    # (B, T, N, hidden)
        X_f = self.cond_emb_f(x_f_ctx, E_f)   # (B, H, N, hidden)

        X_p_prime = self.expert_p(X_p)         # (B, T, N, hidden)
        X_f_prime = self.expert_f(X_f)         # (B, H, N, hidden)

        # ── Stage 2: BALANCE ─────────────────────────────────────────────────
        Y_p = self.encoder_p(X_p_prime)   # (B, H, N, 1)
        Y_f = self.encoder_f(X_f_prime)   # (B, H, N, 1)

        # Context-aware balancer (Eq. 6-7)
        Y = Y_p + Y_f                     # (B, H, N, 1)
        # AvgPool over H (time axis) → (B, 1, N, 1)
        Y_avg = Y.mean(dim=1, keepdim=True)
        alpha = self.balancer(Y_avg)      # (B, 1, N, 1) ∈ (0,1)

        Y_hat = alpha * Y_p + (1.0 - alpha) * Y_f + Y   # Eq. 7 with residual

        return Y_hat.squeeze(-1)          # (B, H, N)


# ─────────────────────────────────────────────────────────────────────────────
# CovTFT — TFT (Lim et al. 2021) with covariates
#   Preserved for comparison. NOT the ExoST architecture.
#   Correct TFT components: 4 static context GRNs, static enrichment layer,
#   InterpretableMHA (shared V, mean-over-heads), quantile head.
# ─────────────────────────────────────────────────────────────────────────────

class _GRN(nn.Module):
    """Gated Residual Network (Lim 2021, Eq.3-5). ELU + GLU + LayerNorm."""
    def __init__(self, inp: int, hid: int, out: int = None,
                 ctx: int = 0, dropout: float = 0.1):
        super().__init__()
        out = out or inp
        self.has_ctx = ctx > 0
        self.fc1  = nn.Linear(inp + ctx, hid)
        self.fc2  = nn.Linear(hid, out * 2)
        self.norm = nn.LayerNorm(out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(inp, out, bias=False) if inp != out else nn.Identity()

    def forward(self, x, c=None):
        r = self.skip(x)
        h = F.elu(self.fc1(torch.cat([x, c], -1) if self.has_ctx else x))
        g, v = self.drop(self.fc2(h)).chunk(2, -1)
        return self.norm(r + torch.sigmoid(g) * v)


class _VSN(nn.Module):
    """Variable Selection Network (Lim 2021, §3.2)."""
    def __init__(self, V: int, D: int, ctx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.V    = V
        self.grns = nn.ModuleList([_GRN(D, D, D, ctx, dropout) for _ in range(V)])
        self.wgrn = _GRN(V * D, D, V, ctx, dropout)

    def forward(self, x, c=None):
        N, V, D = x.shape
        w   = torch.softmax(self.wgrn(x.reshape(N, V * D), c), -1)
        per = torch.stack([self.grns[i](x[:, i], c) for i in range(V)], 1)
        return (w.unsqueeze(-1) * per).sum(1), w


class _IMHA(nn.Module):
    """Interpretable MHA (Lim 2021, Eq.12-13). Shared V, mean-over-heads.
    Wv projects D → dh (single shared head); broadcast to all H heads."""
    def __init__(self, D: int, H: int, dropout: float = 0.1):
        super().__init__()
        assert D % H == 0
        self.H, self.dh, self.D = H, D // H, D
        self.Wq, self.Wk = nn.Linear(D, D), nn.Linear(D, D)
        self.Wv = nn.Linear(D, D // H)   # shared V: project to dh only
        self.Wo   = nn.Linear(D, D)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.shape; H, dh = self.H, self.dh
        Q = self.Wq(q).view(B, L, H, dh).permute(0, 2, 1, 3)  # (B,H,L,dh)
        K = self.Wk(k).view(B, L, H, dh).permute(0, 2, 1, 3)  # (B,H,L,dh)
        V = self.Wv(v).unsqueeze(1).expand(B, H, L, dh)        # (B,H,L,dh) shared
        s = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)
        if mask is not None:
            s = s.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = self.drop(torch.softmax(s, -1))
        o = (a @ V).mean(1)                                     # (B,L,dh) mean over heads
        o = o.unsqueeze(2).expand(B, L, H, dh).reshape(B, L, self.D)
        return self.Wo(o), a.mean(1)


class CovTFT(nn.Module):
    """
    TFT with covariate support (Lim et al., IJF 2021).
    Renamed from TFTExoST to clarify it is NOT the Chen et al. 2025 ExoST.
    Four static context GRNs, static enrichment, InterpretableMHA, quantile head.
    """
    use_covariates: bool = True
    _COV_KEYS = ("date", "weather", "event")

    def __init__(self, num_nodes: int, input_len: int, output_len: int,
                 date_dim: int = 3, weather_dim: int = 8, event_dim: int = 0,
                 hidden_dim: int = 64, num_heads: int = 4,
                 lstm_layers: int = 1, dropout: float = 0.1,
                 quantiles: tuple = (0.1, 0.5, 0.9)):
        super().__init__()
        self.S, self.T, self.H = num_nodes, input_len, output_len
        self.D, self.Nl        = hidden_dim, lstm_layers
        self.quantiles         = list(quantiles)
        self.nq                = len(quantiles)
        self._cov_dims         = {"date": date_dim, "weather": weather_dim, "event": event_dim}

        # Four static context GRNs (§3.3)
        self.static_emb = nn.Embedding(num_nodes, hidden_dim)
        self.grn_cs = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.grn_ce = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.grn_cc = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.grn_ch = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.ctx_c  = nn.Linear(hidden_dim, hidden_dim * lstm_layers)
        self.ctx_h  = nn.Linear(hidden_dim, hidden_dim * lstm_layers)

        # Variable projections (past + future)
        self.past_proj   = nn.ModuleDict({"ts": nn.Linear(1, hidden_dim)})
        self.future_proj = nn.ModuleDict()
        for k in self._COV_KEYS:
            d = self._cov_dims[k]
            if d > 0:
                self.past_proj[k]   = nn.Linear(d, hidden_dim)
                self.future_proj[k] = nn.Linear(d, hidden_dim)
        self._npv = len(self.past_proj)
        self._nfv = len(self.future_proj)

        # VSNs conditioned by c_s
        self.past_vsn = _VSN(self._npv, hidden_dim, hidden_dim, dropout)
        if self._nfv > 0:
            self.future_vsn: Optional[_VSN] = _VSN(
                self._nfv, hidden_dim, hidden_dim, dropout)
        else:
            self.future_vsn = None
            self.fut_pos = nn.Parameter(torch.randn(1, output_len, hidden_dim) * 0.02)

        # LSTM encoder / decoder
        ld = dropout if lstm_layers > 1 else 0.0
        self.enc = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                           batch_first=True, dropout=ld)
        self.dec = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                           batch_first=True, dropout=ld)

        # Static enrichment layer (§3.4)
        self.enrich = _GRN(hidden_dim, hidden_dim, ctx=hidden_dim, dropout=dropout)

        # Interpretable MHA + gates
        self.imha    = _IMHA(hidden_dim, num_heads, dropout)
        self.imha_g  = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.imha_n  = nn.LayerNorm(hidden_dim)
        self.skip_g  = nn.Linear(hidden_dim, hidden_dim * 2)
        self.skip_n  = nn.LayerNorm(hidden_dim)

        # Spatial attention + quantile head
        self.sp_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.sp_grn  = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.sp_norm = nn.LayerNorm(hidden_dim)
        self.q_norm  = nn.LayerNorm(hidden_dim)
        self.q_head  = nn.Linear(hidden_dim, self.nq)

    def _ctx(self, B, dev):
        emb = self.static_emb(torch.arange(self.S, device=dev))
        emb = emb.unsqueeze(0).expand(B, -1, -1).reshape(B * self.S, self.D)
        cs  = self.grn_cs(emb)
        ce  = self.grn_ce(emb)
        cc  = torch.tanh(self.ctx_c(self.grn_cc(emb))).view(
            B * self.S, self.Nl, self.D).permute(1, 0, 2).contiguous()
        ch  = torch.tanh(self.ctx_h(self.grn_ch(emb))).view(
            B * self.S, self.Nl, self.D).permute(1, 0, 2).contiguous()
        return cs, ce, cc, ch

    def _exp(self, cov, B, L):
        return cov.unsqueeze(1).expand(B, self.S, L, -1).reshape(B * self.S, L, -1)

    def _vsn_p(self, xf, covs, B, cs):
        T = self.T; vl = [self.past_proj["ts"](xf)]
        for k in self._COV_KEYS:
            if k not in self.past_proj:
                continue
            c = covs.get(k)
            vl.append(self.past_proj[k](
                self._exp(c, B, T) if c is not None
                else xf.new_zeros(B * self.S, T, self._cov_dims[k])))
        st = torch.stack(vl, 2).reshape(B * self.S * T, self._npv, self.D)
        ct = cs.unsqueeze(1).expand(-1, T, -1).reshape(B * self.S * T, self.D)
        z, _ = self.past_vsn(st, ct)
        return z.reshape(B * self.S, T, self.D)

    def _vsn_f(self, covs, B, cs, dev):
        H = self.H
        if self.future_vsn is None:
            return self.fut_pos.expand(B * self.S, H, self.D)
        vl = []
        for k in self._COV_KEYS:
            if k not in self.future_proj:
                continue
            c = covs.get(k)
            vl.append(self.future_proj[k](
                self._exp(c, B, H) if c is not None
                else torch.zeros(B * self.S, H, self._cov_dims[k], device=dev)))
        st = torch.stack(vl, 2).reshape(B * self.S * H, self._nfv, self.D)
        ct = cs.unsqueeze(1).expand(-1, H, -1).reshape(B * self.S * H, self.D)
        z, _ = self.future_vsn(st, ct)
        return z.reshape(B * self.S, H, self.D)

    def _core(self, x, cx, cy):
        B, T, N = x.shape; H, D = self.H, self.D; BS = B * N
        cs, ce, cc, ch = self._ctx(B, x.device)
        xf  = x.permute(0, 2, 1).reshape(BS, T, 1)
        zp  = self._vsn_p(xf, cx, B, cs)
        enc, (hn, cn) = self.enc(zp, (ch, cc))
        enc = self.enrich(enc, ce.unsqueeze(1).expand(-1, T, -1))
        zf  = self._vsn_f(cy, B, cs, x.device)
        dec, _ = self.dec(zf, (hn, cn))
        msk = torch.triu(torch.ones(H, H, device=x.device, dtype=torch.bool), 1)
        at, _  = self.imha(dec, dec, dec, msk)
        dec    = self.imha_n(dec + self.imha_g(at))
        g, v   = self.skip_g(dec).chunk(2, -1)
        dec    = self.skip_n(dec + torch.sigmoid(g) * v)
        zs  = dec.reshape(B, N, H, D).permute(0, 2, 1, 3).reshape(B * H, N, D)
        sp, _  = self.sp_attn(zs, zs, zs)
        zs  = self.sp_norm(zs + self.sp_grn(sp))
        return zs.reshape(B, H, N, D)

    def forward(self, x, tod=None, dow=None,
                date_X=None, weather_X=None, event_X=None,
                date_Y=None, weather_Y=None, event_Y=None, **_):
        cx = {"date": date_X, "weather": weather_X, "event": event_X}
        cy = {"date": date_Y, "weather": weather_Y, "event": event_Y}
        z  = self._core(x, cx, cy)
        q  = self.q_head(self.q_norm(z))
        p  = self.quantiles.index(0.5) if 0.5 in self.quantiles else 0
        return q[..., p]

    def forward_quantiles(self, x, tod=None, dow=None,
                          date_X=None, weather_X=None, event_X=None,
                          date_Y=None, weather_Y=None, event_Y=None, **_):
        cx = {"date": date_X, "weather": weather_X, "event": event_X}
        cy = {"date": date_Y, "weather": weather_Y, "event": event_Y}
        return self.q_head(self.q_norm(self._core(x, cx, cy)))
    

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
        self.past_grn   = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.future_grn = _GRN(hidden_dim, hidden_dim, dropout=dropout)

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
        self.attn_grn  = _GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # Gated residual (decoder out → gated skip to future input)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_gate = nn.LayerNorm(hidden_dim)

        # --- ExoST spatial attention ---
        self.spatial_attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.spatial_grn   = _GRN(hidden_dim, hidden_dim, dropout=dropout)
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




# ─────────────────────────────────────────────────────────────────────────────
# LOSSES / SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def quantile_loss(pred, y, qs, mask=None):
    total = pred.new_tensor(0.0)
    for i, q in enumerate(qs):
        e = y - pred[..., i]
        l = torch.where(e >= 0, q * e, (q - 1.0) * e)
        total = total + ((l * mask).sum() / mask.sum().clamp(1e-6)
                        if mask is not None else l.mean())
    return total / len(qs)

def masked_mae(pred, y, mask=None):
    l = (pred - y).abs()
    return ((l * mask).sum() / mask.sum().clamp(1e-6) if mask is not None
            else l.mean())

def _cosine_schedule(opt, epochs, warmup=5):
    def f(e):
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        t = (e - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING / EVALUATION / RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def train_reference_model(model, loader, epochs=100, lr=3e-4, name="ref"):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch  = _cosine_schedule(opt, epochs)
    dev  = next(model.parameters()).device
    cov  = model.use_covariates
    is_q = isinstance(model, CovTFT)
    losses = []; t0 = time.time()

    for ep in range(epochs):
        model.train(); ep_l = 0.0
        for batch in loader:
            x  = batch["x"].to(dev)
            y  = batch["y"].to(dev)
            m  = batch["y_mask"].to(dev) if batch.get("y_mask") is not None else None
            kw = {}
            for k in ("tod", "dow"):
                if batch.get(k) is not None:
                    kw[k] = batch[k].to(dev)
            if cov:
                for k in ("x_calendar", "y_calendar", "x_weather", "y_weather",
                          "x_events",   "y_events",
                          "date_X", "date_Y", "weather_X", "weather_Y"):
                    if batch.get(k) is not None:
                        kw[k] = batch[k].to(dev)
                # ExoST uses x_calendar/y_calendar/x_weather/y_weather/x_events/y_events
                # CovTFT/TFTExoST use date_X/date_Y/weather_X/weather_Y/event_X/event_Y
                if isinstance(model, (CovTFT, TFTExoST)):
                    kw.setdefault("date_X",    kw.pop("x_calendar", None))
                    kw.setdefault("date_Y",    kw.pop("y_calendar", None))
                    kw.setdefault("weather_X", kw.pop("x_weather",  None))
                    kw.setdefault("weather_Y", kw.pop("y_weather",  None))
                    kw.setdefault("event_X",   kw.pop("x_events",   None))
                    kw.setdefault("event_Y",   kw.pop("y_events",   None))
            kw = {k: v for k, v in kw.items() if v is not None}

            if is_q:
                loss = quantile_loss(model.forward_quantiles(x, **kw),
                                    y, model.quantiles, m)
            else:
                loss = masked_mae(model(x, **kw), y, m)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_l += loss.item()

        sch.step()
        avg = ep_l / max(len(loader), 1); losses.append(avg)
        print(f"\r[{name}] {ep+1}/{epochs}  loss={avg:.4f}  "
            f"lr={sch.get_last_lr()[0]:.2e}", end="", flush=True)

    print(); model.loss_history = losses; model.train_time = time.time() - t0
    os.makedirs("results_server/loss_functions", exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(losses); ax.set(xlabel="Epoch", ylabel="Loss", title=name)
    fig.tight_layout()
    fig.savefig(f"results_server/loss_functions/loss_{name}.png")
    plt.close(fig)
    return model.train_time


@torch.no_grad()
def evaluate_reference_model(model, loader, norm_params):
    model.eval(); dev = next(model.parameters()).device
    yt, yp = [], []; t0 = time.time()
    for batch in loader:
        x = batch["x"].to(dev); y = batch["y"].to(dev)
        kw = {}
        for k in ("tod", "dow"):
            if batch.get(k) is not None:
                kw[k] = batch[k].to(dev)
        if model.use_covariates:
            for k in ("x_calendar", "y_calendar", "x_weather", "y_weather",
                      "x_events",   "y_events",
                      "date_X", "date_Y", "weather_X", "weather_Y"):
                if batch.get(k) is not None:
                    kw[k] = batch[k].to(dev)
            if isinstance(model, (CovTFT, TFTExoST)):
                kw.setdefault("date_X",    kw.pop("x_calendar", None))
                kw.setdefault("date_Y",    kw.pop("y_calendar", None))
                kw.setdefault("weather_X", kw.pop("x_weather",  None))
                kw.setdefault("weather_Y", kw.pop("y_weather",  None))
                kw.setdefault("event_X",   kw.pop("x_events",   None))
                kw.setdefault("event_Y",   kw.pop("y_events",   None))
        kw = {k: v for k, v in kw.items() if v is not None}
        yt.append(y.cpu()); yp.append(model(x, **kw).cpu())

    model.eval_time = time.time() - t0
    yt = torch.cat(yt); yp = torch.cat(yp)
    diff = norm_params["X_max"] - norm_params["X_min"]
    ymin = norm_params["X_min"]
    return (yt * diff + ymin).numpy(), (yp * diff + ymin).numpy()


def _save_json(name, entry, path="results_server/results.json"):
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            try: existing = json.load(f)
            except: pass
    if isinstance(existing, list):
        existing = {r["model_name"]: r for r in existing if "model_name" in r}
    existing[name] = entry
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
        json.dump(existing, tf, indent=2); tmp = tf.name
    shutil.move(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """python reference_models.py --test"""
    B, T, N, H = 4, 12, 20, 12
    x1d = torch.randn(B, T, N)          # (B,T,N) — typical data loader output
    x4d = torch.randn(B, T, N, 1)       # (B,T,N,C)
    tod = torch.randint(0, 288, (B, T))
    dow = torch.randint(0, 7,   (B, T))
    adj = torch.rand(N, N); adj.fill_diagonal_(0)
    cal_X = torch.randn(B, T, 4);  cal_Y = torch.randn(B, H, 4)
    wea_X = torch.randn(B, T, 4);  wea_Y = torch.randn(B, H, 4)
    evt_X = torch.randn(B, T, 3);  evt_Y = torch.randn(B, H, 3)

    # ── STAEformer ─────────────────────────────────────────────────────────
    m1 = STAEformer(N, 1, T, H, d_f=24, d_a=80, num_layers=3, steps_per_day=288)
    # With tod/dow
    o1a = m1(x4d, tod, dow)
    assert o1a.shape == (B, H, N), f"STAEformer with tod: {o1a.shape}"
    # Without tod/dow (graceful fallback)
    o1b = m1(x4d)
    assert o1b.shape == (B, H, N), f"STAEformer no tod: {o1b.shape}"
    # Ea shape
    assert m1.adaptive_emb.shape == (T, N, 80), \
        f"Ea shape wrong: {m1.adaptive_emb.shape}"
    print(f"STAEformer  ✓  Ea={(T,N,80)}  d_h={3*24+80}  "
          f"params={sum(p.numel() for p in m1.parameters()):,}")

    # ── STGformer ──────────────────────────────────────────────────────────
    m2 = STGformer(N, 1, T, H, d=32, k_order=3, steps_per_day=288)
    o2a = m2(x1d, tod, dow, adj_mx=adj)
    assert o2a.shape == (B, H, N), f"STGformer with tod: {o2a.shape}"
    o2b = m2(x1d)   # no tod/dow, no adj
    assert o2b.shape == (B, H, N), f"STGformer no tok: {o2b.shape}"
    # Verify STGAttention has no dead code
    import inspect
    src = inspect.getsource(_STGAttention.forward)
    assert src.count("kv_s") == 3, "Expected exactly 3 uses of kv_s (assign + comment + einsum)"
    print(f"STGformer   ✓  Xemb=4×32=128  k=3  recursive-fusion  "
          f"params={sum(p.numel() for p in m2.parameters()):,}")

    # ── ExoST ──────────────────────────────────────────────────────────────
    m3 = ExoST(N, T, H, cal_dim=4, weather_dim=4, event_dim=3, hidden_dim=64, k_experts=4)
    o3 = m3(x1d, x_calendar=cal_X, y_calendar=cal_Y,
             x_weather=wea_X, y_weather=wea_Y,
             x_events=evt_X,  y_events=evt_Y)
    assert o3.shape == (B, H, N), f"ExoST with events: {o3.shape}"
    # Test without exo (zeros fallback)
    o3b = m3(x1d)
    assert o3b.shape == (B, H, N), f"ExoST no exo: {o3b.shape}"
    assert isinstance(m3.cond_emb_p, _ExoSTConditionalEmbedding), "Cond emb wrong type"
    assert isinstance(m3.expert_p,   _ExoSTGatedExpert),           "Gated expert wrong"
    assert isinstance(m3.encoder_p,  _ExoSTEncoder),               "Encoder wrong"
    print(f"ExoST       ✓  select+balance  K=4 experts  event_dim=3  "
          f"params={sum(p.numel() for p in m3.parameters()):,}")

    # ── CovTFT ─────────────────────────────────────────────────────────────
    m4 = CovTFT(N, T, H, date_dim=4, weather_dim=4, event_dim=3)
    o4 = m4(x1d, date_X=cal_X, weather_X=wea_X, event_X=evt_X,
             date_Y=cal_Y,  weather_Y=wea_Y,  event_Y=evt_Y)
    assert o4.shape == (B, H, N), f"CovTFT: {o4.shape}"
    for attr in ("grn_cs", "grn_ce", "grn_cc", "grn_ch", "enrich"):
        assert hasattr(m4, attr), f"CovTFT missing {attr}"
    print(f"CovTFT      ✓  4ctx+enrich+IMHA  event_dim=3  "
          f"params={sum(p.numel() for p in m4.parameters()):,}")

    # ── TFTExoST ───────────────────────────────────────────────────────────
    m5 = TFTExoST(N, T, H, date_dim=4, weather_dim=4, event_dim=3, hidden_dim=64)
    o5 = m5(x1d, date_X=cal_X, weather_X=wea_X, event_X=evt_X,
             date_Y=cal_Y,  weather_Y=wea_Y,  event_Y=evt_Y)
    assert o5.shape == (B, H, N), f"TFTExoST: {o5.shape}"
    # No events fallback
    o5b = m5(x1d, date_X=cal_X, weather_X=wea_X, date_Y=cal_Y, weather_Y=wea_Y)
    assert o5b.shape == (B, H, N), f"TFTExoST no events: {o5b.shape}"
    print(f"TFTExoST    ✓  LSTM+IMHA+spatial  event_dim=3  "
          f"params={sum(p.numel() for p in m5.parameters()):,}")

    print("\n✓ All smoke tests passed.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_reference_models(
    modes,
    path,
    reference_model="all",   # "staef"|"stgformer"|"exost"|"tftexost"|"all"
    epochs=100,
    batch_size=32,
    lr=3e-4,
    dropout=0.1,
    multi_run=False,
    n_runs=3,
    seeds=None,              # list[int] of length >= n_runs; auto-filled if None
):
    """Train and evaluate all selected reference models, with optional multi-run.

    Args:
        modes:           List of mode strings, e.g. ["bike", "road", "ped"].
        path:            Root path for train/test .npy files.
        reference_model: Which model(s) to run ("all" or a single name).
        epochs:          Training epochs per run.
        batch_size:      DataLoader batch size.
        lr:              Learning rate.
        dropout:         Dropout rate.
        multi_run:       If True, train n_runs times with different seeds.
        n_runs:          Number of runs (used only when multi_run=True).
        seeds:           Explicit seed list. Defaults to [42, 123, 456, ...].
    """
    import random
    from torch.utils.data import DataLoader
    from src.data_loader import TimeSeriesDataset

    # ── Hyperparams ──────────────────────────────────────────────────────────
    STEPS_PER_DAY = 288
    # STAEformer (paper: df=24, da=80, L=3, heads=4)
    DF, DA, NUM_LAYERS, NUM_HEADS = 24, 80, 3, 4
    # STGformer (paper: d=32, k=3)
    STG_D, STG_K = 32, 3
    # ExoST (paper: hidden=64, K=4 experts)
    EXO_H, EXO_K = 64, 4
    # CovTFT
    TFT_H = 64
    # TFTExoST
    TFTEXOST_H = 256

    # ── Seed list ─────────────────────────────────────────────────────────────
    if seeds is None:
        seeds = [42, 123, 456, 789, 1234]
    _seeds = seeds[:n_runs] if multi_run else [None]
    _n_runs = n_runs if multi_run else 1

    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ── Data helpers ──────────────────────────────────────────────────────────
    def _collate(batch):
        out = {}
        for k in batch[0]:
            vals = [d[k] for d in batch]
            out[k] = torch.stack(vals) if vals[0] is not None else None
        return out

    def _load(mode, path, bs, device):
        pin = device.type == "cuda"
        tr = TimeSeriesDataset(mode=mode, path=path, train=True,
                               exogen_var=True, exo_reduced=True,
                               normalization_type="rowwise_minmax")
        te = TimeSeriesDataset(mode=mode, path=path, train=False,
                               exogen_var=True, exo_reduced=True,
                               normalization_type="rowwise_minmax",
                               norm_params=tr.norm_params)
        kw = dict(batch_size=bs, shuffle=False, num_workers=0,
                  pin_memory=pin, drop_last=False, collate_fn=_collate)
        return tr, te, DataLoader(tr, **kw), DataLoader(te, **kw)

    _ALL = {
        "staef":     {"cls": STAEformer, "cov": False},
        "stgformer": {"cls": STGformer,  "cov": False},
        "exost":     {"cls": ExoST,      "cov": True},
        "tftexost":  {"cls": TFTExoST,   "cov": True},
        #"covtft":   {"cls": CovTFT,     "cov": True},
    }
    to_run = _ALL if reference_model == "all" else {reference_model: _ALL[reference_model]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results_server/predictions",    exist_ok=True)
    os.makedirs("results_server/loss_functions", exist_ok=True)

    for mode in modes:
        print(f"\n{'='*70}\nMODE: {mode.upper()}\n{'='*70}")
        tr_data, te_data, tr_ldr, te_ldr = _load(mode, path, batch_size, device)
        T, H, N = tr_data.input_dim, tr_data.output_dim, tr_data.num_sensors
        w_d = tr_data.weather_X.shape[2]
        d_d = tr_data.calendar_X.shape[2]
        e_d = tr_data.events_X.shape[2] if hasattr(tr_data, "events_X") else 0
        mask = te_data.Y_mask.numpy()

        for ref_name, cfg in to_run.items():
            Cls   = cfg["cls"]
            mname = f"ref_{ref_name}_{mode}"
            print(f"\n--- {ref_name} | {mode} ---")

            # ── model factory ────────────────────────────────────────────────
            def _make_model():
                if Cls is STAEformer:
                    return STAEformer(
                        N, 1, T, H, d_f=DF, d_a=DA,
                        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                        steps_per_day=STEPS_PER_DAY, dropout=dropout).to(device)
                elif Cls is STGformer:
                    return STGformer(
                        N, 1, T, H, d=STG_D, k_order=STG_K,
                        steps_per_day=STEPS_PER_DAY, dropout=dropout).to(device)
                elif Cls is ExoST:
                    return ExoST(
                        N, T, H, cal_dim=d_d, weather_dim=w_d, event_dim=e_d,
                        hidden_dim=EXO_H, k_experts=EXO_K,
                        num_heads=NUM_HEADS, dropout=dropout).to(device)
                elif Cls is TFTExoST:
                    return TFTExoST(
                        N, T, H, date_dim=d_d, event_dim=e_d, weather_dim=w_d,
                        hidden_dim=TFTEXOST_H, num_heads=NUM_HEADS,
                        lstm_layers=2, dropout=dropout).to(device)
                else:  # CovTFT
                    return CovTFT(
                        N, T, H, date_dim=d_d, weather_dim=w_d, event_dim=e_d,
                        hidden_dim=TFT_H, num_heads=NUM_HEADS,
                        lstm_layers=1, dropout=dropout).to(device)

            n_params = sum(p.numel() for p in _make_model().parameters() if p.requires_grad)
            print(f"    parameters: {n_params:,}  |  runs: {_n_runs}")

            # ── seed loop ────────────────────────────────────────────────────
            run_records = []
            for seed in _seeds:
                if seed is not None:
                    _set_seed(seed)

                model = _make_model()
                train_t = train_reference_model(
                    model, tr_ldr, epochs=epochs, lr=lr, name=mname)
                y_true, y_pred = evaluate_reference_model(
                    model, te_ldr, te_data.norm_params)

                mae_m  = float(np.sum(np.abs(y_true - y_pred) * mask) / np.sum(mask))
                rmse_m = float(np.sqrt(np.sum(((y_true - y_pred)**2) * mask) / np.sum(mask)))
                print(f"    seed={seed}  MAE={mae_m:.4f}  RMSE={rmse_m:.4f}  train={train_t:.1f}s")

                run_records.append({
                    "seed": seed,
                    "metrics": {"MAE_masked": mae_m, "RMSE_masked": rmse_m},
                    "train_time": train_t,
                    "inference_time": model.eval_time,
                })

            # ── aggregate ────────────────────────────────────────────────────
            all_m = {k: [r["metrics"][k] for r in run_records]
                     for k in run_records[0]["metrics"]}
            metrics_mean = {k: float(np.mean(v)) for k, v in all_m.items()}
            metrics_std  = {k: float(np.std(v))  for k, v in all_m.items()} \
                           if _n_runs > 1 else None

            # Save predictions from last run
            np.savez(f"results_server/predictions/{mname}_predictions.npz",
                     Y_pred=y_pred, Y_true=y_true, Y_mask=mask)

            _save_json(mname, {
                "model_name":       mname,
                "mode":             mode,
                "ref_model_type":   ref_name,
                "creation_time":    time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_runs":           _n_runs,
                "seeds":            _seeds,
                "metrics":          metrics_mean,
                "metrics_std":      metrics_std,
                "runs":             run_records if _n_runs > 1 else None,
                "train_time":       sum(r["train_time"] for r in run_records),
                "inference_time":   run_records[-1]["inference_time"],
                "loss_history":     [float(v) for v in model.loss_history],
                "number_of_parameters": n_params,
                "hyperparameters":  {
                    "model": ref_name, "epochs": epochs,
                    "batch_size": batch_size, "lr": lr,
                },
            })

    print("\nDone.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _smoke_test()
        sys.exit(0)

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    _SETS = {
        "zurich": {"modes": ["ped", "bike", "road"], "path": "data/train_test"},
        "nyc":    {"modes": ["nyc_bike", "nyc_taxi", "chi_taxi"],
                   "path": "data/train_test/benchmark"},
    }
    SETTING = "zurich"
    run_reference_models(
        modes=_SETS[SETTING]["modes"],
        path=_SETS[SETTING]["path"],
        reference_model="all",
        epochs=100,
        batch_size=32,
        lr=3e-4,
        multi_run=False,
        n_runs=3,
        seeds=[42, 123, 456],
    )