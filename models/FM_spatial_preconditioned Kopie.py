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
        
        # Freeze FM parameters
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
    
    def predict(self, x, horizon):
        """
        Predict future values.
        
        NOTE: No @torch.no_grad() - gradients must flow through for input_module training.
        FM parameters won't update (requires_grad=False) but gradients flow through.
        """
        B, T, S = x.shape
        x = x.to("cpu")
        x_flat = x.permute(0, 2, 1).reshape(B * S, T)
        series_list = list(x_flat)
        
        forecasts = self.pipeline.predict(
            inputs=series_list,
            prediction_length=horizon
        )
        
        preds = torch.stack(
            [torch.as_tensor(f).mean(dim=(0, 1)) for f in forecasts]
        )
        preds = preds[:, :horizon]
        preds = preds.view(B, S, horizon).permute(0, 2, 1)
        return preds.to(self.device, non_blocking=True)
    
    def encode_hidden(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Encode to hidden states. No @torch.no_grad() - gradients can flow."""
        B, T, S = x.shape
        x_cpu = x.detach().to("cpu")
        x_flat = x_cpu.permute(0, 2, 1).reshape(B * S, T)
        series_list = [ts for ts in x_flat]

        emb, _ = self.pipeline.embed(series_list)
        emb = torch.stack(emb)

        if pool == "mean":
            z = emb.mean(dim=1)
        elif pool == "last":
            z = emb[:, -1, :]
        else:
            raise ValueError("pool must be 'mean' or 'last'")

        z = z.view(B, S, -1).permute(0, 2, 1).contiguous()
        return z.to(self.device, non_blocking=True)


# ============================================================================
# COVARIATE ENCODERS
# ============================================================================

class DateEncoder(nn.Module):
    def __init__(self, date_dim=3, d_model=128):
        super().__init__()
        self.hour_emb = nn.Embedding(24, 16)
        self.dow_emb = nn.Embedding(7, 8)
        self.month_emb = nn.Embedding(12, 8)
        
        self.register_buffer('hour_freq', torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer('dow_freq', torch.tensor([1.0]))
        self.register_buffer('month_freq', torch.tensor([1.0]))
        
        fourier_dim = (len(self.hour_freq) + len(self.dow_freq) + len(self.month_freq)) * 2
        self.proj = nn.Sequential(
            nn.Linear(16 + 8 + 8 + fourier_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
    
    def forward(self, date_features):
        B, L, K = date_features.shape
        hour = (date_features[:, :, 0] * 23).long()
        dow = (date_features[:, :, 1] * 6).long()
        month = (date_features[:, :, 2] * 11).long()
        
        h_emb = self.hour_emb(hour)
        d_emb = self.dow_emb(dow)
        m_emb = self.month_emb(month)
        
        hour_norm = date_features[:, :, 0:1]
        dow_norm = date_features[:, :, 1:2]
        month_norm = date_features[:, :, 2:3]
        
        hour_rad = 2 * math.pi * hour_norm
        dow_rad = 2 * math.pi * dow_norm
        month_rad = 2 * math.pi * month_norm
        
        hour_fourier = torch.cat([
            torch.sin(hour_rad * f) for f in self.hour_freq
        ] + [
            torch.cos(hour_rad * f) for f in self.hour_freq
        ], dim=-1)
        
        dow_fourier = torch.cat([
            torch.sin(dow_rad * f) for f in self.dow_freq
        ] + [
            torch.cos(dow_rad * f) for f in self.dow_freq
        ], dim=-1)
        
        month_fourier = torch.cat([
            torch.sin(month_rad * f) for f in self.month_freq
        ] + [
            torch.cos(month_rad * f) for f in self.month_freq
        ], dim=-1)
        
        all_features = torch.cat([
            h_emb, d_emb, m_emb,
            hour_fourier, dow_fourier, month_fourier
        ], dim=-1)
        
        return self.proj(all_features)


class EventEncoder(nn.Module):
    def __init__(self, event_dim, d_model=128):
        super().__init__()
        self.event_dim = event_dim
        self.event_proj = nn.Linear(event_dim, d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    
    def forward(self, event_features):
        B, L, K = event_features.shape
        x = self.event_proj(event_features)
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm(x + attn_out)
        return x


class WeatherEncoder(nn.Module):
    def __init__(self, weather_dim, d_model=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(weather_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, weather_features):
        return self.proj(weather_features)


class AttentionFusion(nn.Module):
    def __init__(self, traffic_dim: int, cov_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(traffic_dim, hidden_dim)
        self.k_proj = nn.Linear(cov_dim, hidden_dim)
        self.v_proj = nn.Linear(cov_dim, hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scalar = nn.Linear(hidden_dim, 1)

    def forward(self, traffic: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        B, T, N, D_t = traffic.shape
        
        traffic_flat = traffic.reshape(B * T, N, D_t)
        cov_flat = covariates.reshape(B * T, N, covariates.shape[-1])
        
        Q = self.q_proj(traffic_flat).reshape(B * T * N, 1, self.hidden_dim)
        K = self.k_proj(cov_flat).reshape(B * T * N, 1, self.hidden_dim)
        V = self.v_proj(cov_flat).reshape(B * T * N, 1, self.hidden_dim)
        
        attn_out, _ = self.attn(Q, K, V)
        attn_out = attn_out.reshape(B, T, N, self.hidden_dim)
        
        traffic_proj = self.q_proj(traffic).reshape(B, T, N, self.hidden_dim)
        out = self.norm(attn_out + traffic_proj)
        out = self.output_proj(out)
        
        correction = self.to_scalar(out)
        return (traffic + correction).squeeze(-1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, queries, context):
        attn_out, _ = self.cross_attn(queries, context, context)
        return self.norm(queries + self.scale * attn_out)


class GraphAttentionLayer(nn.Module):
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
        B, S, T, D = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, S, D)
        
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
# INPUT MODULE (Configuration-Driven)
# ============================================================================

class InputModule(nn.Module):
    """
    Configuration-driven input conditioning module.
    Only creates components for covariates that will be used.
    """
    def __init__(
        self,
        num_sensors,
        date_dim=3,
        event_dim=0,
        weather_dim=5,
        use_date=False,
        use_event=False,
        use_weather=False
    ):
        super().__init__()
        
        self.use_date = use_date
        self.use_event = use_event
        self.use_weather = use_weather
        
        # Only create if at least one covariate is used
        if not any([use_date, use_event, use_weather]):
            # No conditioning at all - just identity
            self.conditioner = None
            return
        
        # Create only the encoders we need
        hidden_dim = 128
        self.encoders = nn.ModuleDict()
        
        if use_date:
            self.encoders['date'] = DateEncoder(date_dim, d_model=hidden_dim)
        
        if use_event and event_dim > 0:
            self.encoders['event'] = EventEncoder(event_dim, d_model=hidden_dim)
        
        if use_weather and weather_dim > 0:
            self.encoders['weather'] = WeatherEncoder(weather_dim, d_model=hidden_dim)
        
        # Fusion module
        num_covariates = len(self.encoders)
        if num_covariates > 0:
            cov_dim = num_covariates * hidden_dim
            self.fusion = AttentionFusion(
                traffic_dim=1,
                cov_dim=cov_dim,
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=0.1
            )
            self.num_sensors = num_sensors
            self.hidden_dim = hidden_dim
        else:
            self.fusion = None
    
    def forward(self, x, date_X=None, event_X=None, weather_X=None):
        """
        Forward pass - only processes covariates that were configured.
        
        Args:
            x: (B, T, S) traffic data
            date_X, event_X, weather_X: covariate tensors (can be None)
        
        Returns:
            (B, T, S) conditioned traffic or unchanged x if no conditioning
        """
        # No conditioning configured
        if not hasattr(self, 'fusion') or self.fusion is None:
            return x
        
        B, T, S = x.shape
        device = x.device
        
        # Encode covariates (only those configured)
        encoded_covs = []
        
        if self.use_date and 'date' in self.encoders:
            date_enc = self.encoders['date'](date_X)
            # Expand to all sensors: (B, T, d_model) → (B, T, S, d_model)
            encoded_covs.append(date_enc.unsqueeze(2).expand(-1, -1, S, -1))
        
        if self.use_event and 'event' in self.encoders:
            if event_X is not None:
                event_enc = self.encoders['event'](event_X)
            else:
                # Encoder was initialized but no data provided — use zeros to keep cov_dim consistent
                event_enc = torch.zeros(B, T, self.hidden_dim, device=device)
            encoded_covs.append(event_enc.unsqueeze(2).expand(-1, -1, S, -1))
        
        if self.use_weather and 'weather' in self.encoders:
            weather_enc = self.encoders['weather'](weather_X)
            encoded_covs.append(weather_enc.unsqueeze(2).expand(-1, -1, S, -1))
        
        if len(encoded_covs) == 0:
            return x
        
        # Concatenate all encoded covariates
        cov_cat = torch.cat(encoded_covs, dim=-1)  # (B, T, S, cov_dim)
        
        # Fuse with traffic
        x_unsqueezed = x.unsqueeze(-1)  # (B, T, S, 1)
        x_conditioned = self.fusion(x_unsqueezed, cov_cat)
        
        return x_conditioned


# ============================================================================
# PREDICTION MODULE (Configuration-Driven)
# ============================================================================

class PredictionModule(nn.Module):
    """
    Configuration-driven prediction module.
    Only creates components for covariates and features that will be used.
    """
    def __init__(
        self,
        num_sensors,
        output_len,
        date_dim=3,
        event_dim=0,
        weather_dim=5,
        fm_hidden_dim=6144,
        hidden_dim=256,
        graph_layers=2,
        use_date=False,
        use_event=False,
        use_weather=False,
        device=None
    ):
        super().__init__()
        self.device = device
        self.num_sensors = num_sensors
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        
        self.use_date = use_date
        self.use_event = use_event
        self.use_weather = use_weather
        
        # FM adapter (always needed)
        self.fm_adapter = nn.Sequential(
            nn.Linear(fm_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Learned adjacency matrix
        self.adj_logits = nn.Parameter(torch.randn(num_sensors, num_sensors) * 0.1)
        
        # Positional encoding
        self.spatial_pos = nn.Parameter(torch.randn(1, num_sensors, 1, hidden_dim))
        
        # Spatial graph layers (always needed)
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads=8) for _ in range(graph_layers)
        ])
        
        # Future projection (always needed)
        self.future_proj = nn.Linear(hidden_dim, output_len * hidden_dim)
        
        # Future covariate encoders and fusion (only if used)
        self.future_encoders = nn.ModuleDict()
        self.future_fusions = nn.ModuleDict()
        
        if use_date:
            self.future_encoders['date'] = DateEncoder(date_dim, d_model=128)
            self.future_fusions['date'] = CrossAttentionFusion(hidden_dim, num_heads=8)
        
        if use_event and event_dim > 0:
            self.future_encoders['event'] = EventEncoder(event_dim, d_model=128)
            self.future_fusions['event'] = CrossAttentionFusion(hidden_dim, num_heads=8)
        
        if use_weather and weather_dim > 0:
            self.future_encoders['weather'] = WeatherEncoder(weather_dim, d_model=128)
            self.future_fusions['weather'] = CrossAttentionFusion(hidden_dim, num_heads=8)
        
        # Covariate projection (only if any future covariates used)
        if len(self.future_encoders) > 0:
            self.covariate_proj = nn.Linear(128, hidden_dim)
        
        # Prediction head (always needed)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_adjacency(self):
        """Get learned adjacency matrix"""
        return torch.sigmoid(self.adj_logits)
    
    def forward(self, fm_hidden, date_Y=None, event_Y=None, weather_Y=None):
        """
        Forward pass - only processes configured future covariates.
        
        Args:
            fm_hidden: (B, D, S) - FM hidden states
            date_Y, event_Y, weather_Y: future covariates (can be None)
        
        Returns:
            adapter_corrections: (B, H, S)
        """
        B, D, S = fm_hidden.shape
        H = self.output_len
        
        # Project FM hidden states
        fm_hidden = fm_hidden.transpose(1, 2)  # (B, S, D)
        Z = self.fm_adapter(fm_hidden)  # (B, S, hidden_dim)
        
        Z = Z.unsqueeze(2)  # (B, S, 1, hidden_dim)
        Z = Z + self.spatial_pos
        
        # Spatial graph layers
        adj = self.get_adjacency()
        for graph_layer in self.graph_layers:
            Z = graph_layer(Z, adj=adj)
        
        # Project to future
        Z_spatial = Z.squeeze(2)
        Z_future_flat = self.future_proj(Z_spatial)
        Z_future = Z_future_flat.view(B, S, H, self.hidden_dim)
        
        # Future covariate fusion (only for configured covariates)
        Z_future_reshaped = Z_future.reshape(B * S, H, self.hidden_dim)
        
        if self.use_date and 'date' in self.future_encoders:
            date_future = self.future_encoders['date'](date_Y)
            date_future = date_future.to(self.device, non_blocking=True)
            date_future = self.covariate_proj(date_future)
            date_future_exp = date_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_fusions['date'](Z_future_reshaped, date_future_exp)
        
        if self.use_event and 'event' in self.future_encoders and event_Y is not None:
            event_future = self.future_encoders['event'](event_Y)
            event_future = event_future.to(self.device, non_blocking=True)
            event_future = self.covariate_proj(event_future)
            event_future_exp = event_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_fusions['event'](Z_future_reshaped, event_future_exp)
        
        if self.use_weather and 'weather' in self.future_encoders:
            weather_future = self.future_encoders['weather'](weather_Y)
            weather_future = weather_future.to(self.device, non_blocking=True)
            weather_future = self.covariate_proj(weather_future)
            weather_future_exp = weather_future.unsqueeze(1).expand(B, S, H, self.hidden_dim).reshape(B * S, H, self.hidden_dim)
            Z_future_reshaped = self.future_fusions['weather'](Z_future_reshaped, weather_future_exp)
        
        Z_future = Z_future_reshaped.view(B, S, H, self.hidden_dim)
        
        # Prediction corrections
        adapter_corrections = self.pred_head(Z_future).squeeze(-1)
        adapter_corrections = adapter_corrections.transpose(1, 2)  # (B, H, S)
        
        return adapter_corrections


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model_from_config(
    config,
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
    device=None
):
    """
    Factory function to create models based on configuration.
    
    Args:
        config: dict with keys:
            - use_past_date, use_past_event, use_past_weather
            - use_future_date, use_future_event, use_future_weather
        
    Returns:
        tuple: (input_module, prediction_module) or (None, prediction_module)
    """
    # Extract flags
    use_past_date = config.get('use_past_date', False)
    use_past_event = config.get('use_past_event', False)
    use_past_weather = config.get('use_past_weather', False)
    
    use_future_date = config.get('use_future_date', False)
    use_future_event = config.get('use_future_event', False)
    use_future_weather = config.get('use_future_weather', False)
    
    # Create input module (only if any past covariates used)
    if any([use_past_date, use_past_event, use_past_weather]):
        input_module = InputModule(
            num_sensors=num_sensors,
            date_dim=date_dim,
            event_dim=event_dim,
            weather_dim=weather_dim,
            use_date=use_past_date,
            use_event=use_past_event,
            use_weather=use_past_weather
        )
    else:
        input_module = None
    
    # Create prediction module (always create, but configure future covariates)
    prediction_module = PredictionModule(
        num_sensors=num_sensors,
        output_len=output_len,
        date_dim=date_dim,
        event_dim=event_dim,
        weather_dim=weather_dim,
        fm_hidden_dim=fm_hidden_dim,
        hidden_dim=hidden_dim,
        graph_layers=graph_layers,
        use_date=use_future_date,
        use_event=use_future_event,
        use_weather=use_future_weather,
        device=device
    )
    
    return input_module, prediction_module


# ============================================================================
# CONTAINER MODELS
# ============================================================================

class FullModel(nn.Module):
    """
    Complete model container.
    
    Architecture: X → InputModule? → FM → PredictionModule → predictions
    """
    def __init__(self, fm_encoder, input_module, prediction_module, output_len, device=None):
        super().__init__()
        self.device = device
        self.fm_encoder = fm_encoder
        self.output_len = output_len
        
        self.input_module = input_module  # Can be None
        self.prediction_module = prediction_module  # Always present
    
    def forward(self, x, date_X=None, event_X=None, weather_X=None, 
                date_Y=None, event_Y=None, weather_Y=None, return_fm_preds=False):
        """
        Full forward pass.
        
        Returns: (B, H, S) or (predictions, fm_preds) if return_fm_preds=True
        """
        H = self.output_len
        
        # Input conditioning (if module exists)
        if self.input_module is not None:
            X_conditioned = self.input_module(x, date_X, event_X, weather_X)
        else:
            X_conditioned = x
        
        # FM predictions and hidden states
        fm_preds = self.fm_encoder.predict(X_conditioned, H)
        fm_hidden = self.fm_encoder.encode_hidden(X_conditioned, pool='mean')
        
        # Prediction module
        adapter_corrections = self.prediction_module(fm_hidden, date_Y, event_Y, weather_Y)
        
        predictions = fm_preds + adapter_corrections
        
        if return_fm_preds:
            return predictions, fm_preds
        
        return predictions


class Stage1Model(nn.Module):
    """
    Stage 1 model: Train input conditioning to optimize FM predictions.
    
    Architecture: X → InputModule → FM → predictions
    """
    def __init__(self, fm_encoder, input_module, output_len, device=None):
        super().__init__()
        self.device = device
        self.fm_encoder = fm_encoder
        self.output_len = output_len
        self.input_module = input_module
    
    def forward(self, x, date_X=None, event_X=None, weather_X=None):
        """Returns: FM predictions after input conditioning"""
        X_conditioned = self.input_module(x, date_X, event_X, weather_X)
        fm_preds = self.fm_encoder.predict(X_conditioned, self.output_len)
        return fm_preds


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(model, test_loader, norm_params, is_stage1=False):
    """Evaluate model and denormalize"""
    model.eval()
    device = next(model.parameters()).device
    
    y_true_list, y_pred_list = [], []
    start_time = time.time()
    
    for batch in test_loader:
        x = batch["x"].to(device)
        y_true = batch["y"].to(device)
        
        date_X = batch["x_calendar"].to(device)
        date_Y = batch["y_calendar"].to(device)
        event_X = batch["x_events"].to(device) if "x_events" in batch else None
        event_Y = batch["y_events"].to(device) if "y_events" in batch else None
        weather_X = batch["x_weather"].to(device)
        weather_Y = batch["y_weather"].to(device)
        
        if is_stage1:
            y_pred = model(x, date_X, event_X, weather_X)
        else:
            y_pred = model(x, date_X, event_X, weather_X, date_Y, event_Y, weather_Y)
        
        y_true_list.append(y_true.cpu())
        y_pred_list.append(y_pred.cpu())
    
    eval_time = time.time() - start_time
    
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    
    diff = norm_params['X_max'] - norm_params['X_min']
    y_min = norm_params['X_min']
    y_true_denorm = y_true * diff + y_min
    y_pred_denorm = y_pred * diff + y_min
    
    return y_true_denorm.numpy(), y_pred_denorm.numpy(), eval_time