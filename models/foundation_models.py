#from chronos.forecasting import ChronosPipeline
#from timesfm import TimesFm
from transformers import AutoModel, AutoConfig
from chronos import Chronos2Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import math


class Chronos2Wrapper:
    def __init__(self):
        self.name = "Chronos-2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.device)
        
        self.pipeline.model.eval()
        
    
    @torch.no_grad()
    def predict(self, x, horizon, return_latents=False):
        """
        x: torch.Tensor of shape (B, T, S)
        horizon: int
        returns: torch.Tensor of shape (B, H, S)
        """
        B, T, S = x.shape
        
        # torch tensor b,h,s to b*s, h list
        x = x.to(self.device)

        # Reshape to (B*S, T)
        x_flat = x.permute(0, 2, 1).reshape(B * S, T)

        # Convert to list of 1D tensors (required by Chronos)
        series_list = list(x_flat)

        # Single batched Chronos call
        forecasts = self.pipeline.predict(
            inputs=series_list,
            prediction_length=horizon
        )

        # Stack and reshape back
        preds = torch.stack(
            [torch.as_tensor(f).mean(dim=(0, 1)) for f in forecasts]
        )  # (B*S, H)
        # only keep first and last dimension of preds
        preds = preds[:, :horizon]

        # Reshape to (B, S, H) → (B, H, S)
        preds = preds.view(B, S, horizon).permute(0, 2, 1)
        
        # If latents are requested, return them as well
        if return_latents:
            
            latents = torch.stack(
                [torch.as_tensor(forecast) for forecast in forecasts]
            )
            latents = latents.view(B, S, -1)  # (B, S, latent_dim)
            latents = latents.permute(0, 2, 1)  # (B, latent_dim, S)
            
            return preds, latents
        
        return preds, None



class Chronos2WrapperConditional:
    def __init__(self, input_dim, exo_dim, hidden_dim=64, dropout=0.1):
        self.name = "Chronos-2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.device)
        
        #self.pipeline.model.eval()
        
        self.encoder = ConditionalEncoder(input_dim, exo_dim, hidden_dim, dropout).to(self.device)
        
    @torch.no_grad()
    def predict(self, x, horizon, exo, return_latents=False):
        return self._predict(x, horizon, exo, return_latents)
    
    def _predict(self, x, horizon, exo, return_latents=False):
        """
        x: torch.Tensor of shape (B, T, K)
        horizon: int
        exo: torch.Tensor of shape (B, T, L) or None
        returns: torch.Tensor of shape (B, H, K)
        """
        B, T, K = x.shape
        
        # Reshape x to (B, K, T, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, K, T, 1)
        
        # Reshape exo to (B, K, T, L)
        exo = exo.permute(0, 2, 1).unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, T, L)
        
        # Normalize input using the conditional encoder
        x_norm = self.encoder(x, exo)  # (B, K, T)
        
        # Reshape to (B*K, T)
        x_norm_flat = x_norm.permute(0, 2, 1).reshape(B * K, T)
        
        # Convert to list of 1D tensors (required by Chronos)
        series_list = list(x_norm_flat)
        
        # Single batched Chronos call
        forecasts = self.pipeline.predict(
            inputs=series_list,
            prediction_length=horizon
        )
        
        # Stack and reshape back
        preds = torch.stack(
            [torch.as_tensor(f).mean(dim=(0, 1)) for f in forecasts]
        )  # (B*K, H)
        preds = preds[:, :horizon]
        
        # Reshape to (B, H, K)
        preds = preds.view(B, K, horizon).permute(0, 2, 1)
        
        return preds, None
    
    def train(self, dataloader, optimizer, criterion, epochs=10, horizon=10):
        """ Train the conditional encoder to improve the overall pipeline performance. """
        
        self.encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                exo = batch["x_calendar"].to(self.device)  # batch["x_weather"]
                
                optimizer.zero_grad()
                x.requires_grad = True
                exo.requires_grad = True
                
                # Forward pass
                B, T, K = x.shape
                _, _, L = exo.shape
                
                # Reshape x to (B, K, T, 1)
                x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, K, T, 1)
                
                # Reshape exo to (B, K, T, L)
                exo = exo.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, T, L)
                
                # Normalize input using the conditional encoder
                x_norm = self.encoder(x, exo)  # (B, K, T)
                
                # Reshape to (B*K, T)
                x_norm_flat = x_norm.permute(0, 2, 1).reshape(B * K, T)
                
                # Convert to list of 1D tensors (required by Chronos)
                series_list = list(x_norm_flat)
                
                # Single batched Chronos call
                forecasts = self.pipeline.predict(
                    inputs=series_list,
                    prediction_length=horizon
                )
                
                # Stack and reshape back
                preds = torch.stack(
                    [torch.as_tensor(f).mean(dim=(0, 1)) for f in forecasts]
                )  # (B*K, H)
                preds = preds[:, :horizon]
                
                # Reshape to (B, H, K)
                preds = preds.view(B, K, horizon).permute(0, 2, 1)
                
                # Compute loss
                loss = criterion(preds, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
                
    
class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, exo_dim, hidden_dim=64, dropout=0.1):
        super(ConditionalEncoder, self).__init__()
        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_e = nn.Linear(exo_dim, hidden_dim)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        self.reduce = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, exo):
        # x: (B, K, T, 1)
        # exo: (B, K, T, L)
        #B, T, K = x.shape
        #_, _, L = exo.shape
        #print(x.shape, exo.shape)
        #print(self.W_x.weight.shape, self.W_e.weight.shape)
        emb = self.W_x(x) + self.W_e(exo) + self.b
        emb = F.relu(emb)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        
        emb = self.reduce(emb).squeeze(-1)

        return emb


class TimesFMWrapper:
    def __init__(self):
        self.name = "TimesFM"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_timesfm()

    def init_timesfm(self):
        """Initialize the TimesFM pretrained model."""
        from transformers import TimesFmModelForPrediction  # optional dependency
        # The right class with from_pretrained()
        model = TimesFmModelForPrediction.from_pretrained(
            "google/timesfm-2.0-500m-pytorch",
            #horizon_length=horizon_len
        )
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, x, horizon):
        """
        x: torch.Tensor of shape (B, T, S)
        horizon: int
        returns: torch.Tensor of shape (B, H, S)
        """
        B, T, S = x.shape
        x = x.to(self.device)

        # Build list of 1D tensors per series as expected by HF TimesFM
        inputs = [x[b, :, s] for b in range(B) for s in range(S)]

        # HF TimesFM expects freq tensor (can be dummy zeros if unknown)
        freq = torch.zeros(len(inputs), dtype=torch.long, device=self.device)

        # Run model
        outputs = self.model(
            past_values=inputs,
            freq=freq,
            return_dict=True
        )

        # mean_predictions has shape (B*S, H)
        preds = outputs.mean_predictions
        
         # Slice to requested horizon
        preds = preds[:, :horizon]

        # reshape back to (B, S, H) then permute to (B, H, S)
        preds = preds.view(B, S, -1).permute(0, 2, 1)
        return preds


class TTMWrapper:
    def __init__(self, horizon_len, device=None):
        self.horizon = horizon_len
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.init_ttm()
        self.model.to(self.device)
        self.model.eval()

    def init_ttm(self):
        """
        Initialize IBM Tiny Time Mixer (TTM)
        """
        config = AutoConfig.from_pretrained(
            "ibm/ttm-base",
            prediction_length=self.horizon
        )
        model = AutoModel.from_pretrained(
            "ibm/ttm-base",
            config=config
        )
        return model

    @torch.no_grad()
    def predict(self, x, horizon):
        """
        x: torch.Tensor (B, T, S)
        returns: torch.Tensor (B, H, S)
        """
        B, T, S = x.shape
        x = x.to(self.device)

        # TTM expects (B, S, T)
        x_in = x.permute(0, 2, 1)

        # Forward pass
        outputs = self.model(x_in)

        # Output shape: (B, S, H)
        y_hat = outputs.prediction

        # Convert to (B, H, S)
        return y_hat.permute(0, 2, 1)
    
class ErrorCorrectionModuleOld(nn.Module): ################################################
    def __init__(self, horizon, node_dim, latent_dim=None, ext_dim=None, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.node_dim = node_dim

        self.use_latents = latent_dim is not None
        self.use_external = ext_dim is not None

        # --- graph attention (always on) ---
        self.q = nn.Linear(horizon, hidden_dim)
        self.k = nn.Linear(horizon, hidden_dim)
        self.v = nn.Linear(horizon, hidden_dim)

        # --- external conditioning (optional) ---
        """
        if self.use_external:
            self.reduce_ext = nn.Linear(ext_dim, 1)
            self.film = nn.Linear(ext_dim, 2 * hidden_dim)
        
        if self.use_external:
            # maps arbitrary (K,L) to hidden_dim FiLM
            self.ext_mlp = nn.Sequential(
                nn.Flatten(start_dim=1),    # (B, K*L)
                nn.Linear(ext_dim, hidden_dim * 2)  # gamma + beta
            )
        """
        self.use_external = True if ext_dim is not None else False
            
        self.ext_to_node = nn.Linear(self.hidden_dim, horizon * node_dim)

        # --- residual head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, base_preds, latents=None, external=None):
        # base_preds: (B, H, N)
        base_preds = base_preds.permute(0, 2, 1)  # (B, N, H)

        # 1) build node features (always)
        if self.use_latents and latents is not None:
            nodes = torch.cat([base_preds, latents], dim=1)
            print(nodes.shape)
        else:
            nodes = base_preds
            
        if external is not None:
            # external: (B, K, L)
            B = base_preds.shape[0]
            # get the most important features of external data (K = number of external features, L = time steps)
            # reduce to a general embedding of conditions into 2D
            
            ext_flat = external.flatten(start_dim=1)
            ext_encoder = nn.Sequential(
                nn.Linear(ext_flat.shape[1], self.hidden_dim),
                nn.ReLU()
            ).to(base_preds.device)
            ext_emb = ext_encoder(ext_flat)  # (B, E)
            node_bias = self.ext_to_node(ext_emb)       # (B, H * node_dim)
            node_bias = node_bias.view(B, self.node_dim, self.horizon)  # (B, H, node_dim)
            #node_bias = node_bias.permute(0, 2, 1)  # (B, N, H)
            nodes = nodes + node_bias
            

        # 2) graph attention (always)
        Q = self.q(nodes)
        K = self.k(nodes)
        V = self.v(nodes)

        attn = torch.softmax(
            Q @ K.transpose(-1, -2) / Q.size(-1)**0.5,
            dim=-1
        )
        z = attn @ V  # (B,N,Hd)

        # 4) residual correction
        delta = self.head(z)
        return (base_preds + delta).permute(0, 2, 1)  # (B, H, N)


# New Model Tue Jan 11 2025 ###########################################

class WeatherEncoder(nn.Module):
    """
    Raw weather: (B, K_weather, L)  # K_weather vars, L past steps
    -> (B, d) fixed-size vector that summarises the *future-relevant* weather.
    """
    def __init__(self, k_weather, l_steps, d_model=64, d_out=64):
        super().__init__()
        self.k = k_weather
        self.l = l_steps
        # 1-D CNN across time for each variable → (B, d_model, L')
        self.cnn = nn.Conv1d(k_weather, d_model, kernel_size=3, padding=1)
        # attention pool across time
        self.att_pool = nn.Sequential(
            nn.Conv1d(d_model, 1, 1),          # (B,1,L')
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, x):                     # (B, K, L)
        z = self.cnn(x)                       # (B, d_model, L)
        w = self.att_pool(z)                  # (B, 1, L)
        pooled = (z * w).sum(dim=-1)          # (B, d_model)
        return self.proj(pooled)              # (B, d_out)

class CalendarEncoder(nn.Module):
    """
    Calendar/events: (B, K_cal, L)  # K_cal = {month, dow, holiday, special_event, …}
    -> (B, d) vector.
    """
    def __init__(self, k_cal, d_out=64):
        super().__init__()
        self.emb = nn.Linear(k_cal, 64)
        self.rnn = nn.GRU(64, 64, batch_first=True)
        self.out = nn.Linear(64, d_out)

    def forward(self, x):                     # (B, K_cal, L)
        #x = x.transpose(1, 2)                 # (B, L, K_cal)
        x = self.emb(x)                       # (B, L, 64)
        _, h = self.rnn(x)                    # h: (1, B, 64)
        return self.out(h.squeeze(0))         # (B, d_out)

class ModalityXAttn(nn.Module):
    def __init__(self, d_node, d_ctx, n_heads=4):
        super().__init__()
        self.q = nn.Linear(d_node, d_node)
        self.kv_node = nn.Linear(d_node, 2*d_node)
        self.kv_ctx  = nn.Linear(d_ctx,  2*d_node)
        self.film_node = nn.Linear(d_node, 2*d_node)
        self.film_ctx  = nn.Linear(d_ctx,  2*d_node)
        self.out = nn.Linear(d_node, d_node)
        self.n_heads = n_heads
        self.d = d_node // n_heads

    def forward(self, node, ctx):             # node:(B,N,d)  ctx:(B,d_ctx)
        B, N, _ = node.shape
        ctx_exp = ctx.unsqueeze(1).expand(-1, N, -1)  # (B,N,d_ctx)

        Q  = self.q(node).view(B, N, self.n_heads, self.d).transpose(1,2)  # (B,h,N,d)
        KN, VN = map(lambda t: t.view(B, N, self.n_heads, self.d).transpose(1,2),
                    torch.chunk(self.kv_node(node), 2, -1))
        KC, VC = map(lambda t: t.view(B, N, self.n_heads, self.d).transpose(1,2),
                    torch.chunk(self.kv_ctx(ctx_exp), 2, -1))

        # attention weights
        score_n = (Q @ KN.transpose(-2,-1)) / math.sqrt(self.d)   # (B,h,N,N)
        score_c = (Q @ KC.transpose(-2,-1)) / math.sqrt(self.d)   # (B,h,N,N)
        attn_n = torch.softmax(score_n, dim=-1)
        attn_c = torch.softmax(score_c, dim=-1)

        # FiLM gates
        g_n = self.film_node(node).unsqueeze(1)          # (B,1,N,2d)
        g_c = self.film_ctx(ctx_exp).unsqueeze(1)
        gamma_n, beta_n = g_n.chunk(2, -1)
        gamma_n = gamma_n.view(B, N, self.n_heads, self.d).transpose(1,2)
        beta_n  = beta_n.view(B, N, self.n_heads, self.d).transpose(1,2)
        gamma_c, beta_c = g_c.chunk(2, -2)
        gamma_c = gamma_c.view(B, N, self.n_heads, self.d).transpose(1,2)
        beta_c  = beta_c.view(B, N, self.n_heads, self.d).transpose(1,2)
        
        # VN: 32, 4, 14, 16 - B, h_head, N, pred_window
        # gamma_n: 32, 1, 14, 64

        VN = VN * gamma_n + beta_n # shape (B, B, h, N, d)
        VC = VC * gamma_c + beta_c

        out = attn_n @ VN  +  attn_c @ VC # (B,N,2048)
        out = out.transpose(1,2).contiguous().view(B, N, -1)
        return self.out(out)                              # (B,N,d)

class ErrorCorrectionModule(nn.Module):
    def __init__(self, horizon_in, horizon_out, node_dim, k_weather=5, k_cal=4, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon_in = horizon_in
        self.horizon_out = horizon_out
        self.node_dim  = node_dim

        # --- encoders ---
        self.w_enc = WeatherEncoder(k_weather, horizon_in, d_out=hidden_dim)
        self.c_enc = CalendarEncoder(k_cal, d_out=hidden_dim)

        # --- node feature (same as before) ---
        self.q = nn.Linear(horizon_out, hidden_dim)
        self.k = nn.Linear(horizon_out, hidden_dim)
        self.v = nn.Linear(horizon_out, hidden_dim)

        # --- cross-attn fusion ---
        self.xattn_w = ModalityXAttn(hidden_dim, hidden_dim)
        self.xattn_c = ModalityXAttn(hidden_dim, hidden_dim)

        # --- final residual head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_out),
        )

    def forward(self, base_preds, latents=None, weather=None, calendar=None):
        # base_preds: (B, H, N)  -> (B, N, H)
        base_preds = base_preds.permute(0, 2, 1)
        B, N, _    = base_preds.shape

        # node repr
        nodes = base_preds
        if latents is not None:
            nodes = torch.cat([base_preds, latents], dim=1)

        # self attn baseline
        Q = self.q(nodes); K = self.k(nodes); V = self.v(nodes)
        score = Q @ K.transpose(-2,-1) / math.sqrt(Q.size(-1))
        attn  = torch.softmax(score, dim=-1)
        z     = attn @ V                                          # (B,N,d)

        # modality context
        if weather is not None:
            x_weather = weather[0]
            x_weather = x_weather.permute(0, 2, 1)
            assert torch.isfinite(x_weather).all(), "weather data has NaNs or Infs!"
            # currently only using weather_past -> use both with torch.cat([weather_past, weather_future], dim=-1) before the 1-D CNN
            w_ctx = self.w_enc(x_weather)            # (B, d) -> 32, 64
            z = z + self.xattn_w(z, w_ctx)           # 32, 14, 64
        if calendar is not None:
            x_calendar = calendar[0]
            assert torch.isfinite(x_calendar).all(), "calendar data has NaNs or Infs!"
            c_ctx = self.c_enc(x_calendar)
            z = z + self.xattn_c(z, c_ctx)

        delta = self.head(z)
        return (base_preds + delta).permute(0, 2, 1)   # (B,H,N)


class ExtendedFM(nn.Module):
    def __init__(self, foundation, correction, horizon, mode):
        super().__init__()
        self.foundation = foundation   # Chronos2Wrapper
        self.correction = correction   # nn.Module
        self.horizon = horizon
        self.mode = mode
        self.train_time = 0
        self.eval_time = 0

    def forward(self, x, latents=False, weather=True, calendar=True, no_corrections=False):
        preds, latents = self.foundation.predict(
            x, self.horizon, return_latents=latents
        )

        if no_corrections:
            return preds

        return self.correction(preds, latents, weather, calendar)
    
    def train_correction(self, dataloader, epochs=50, use_weather=True, use_calendar=True, use_latents=False, model_name=None):
        #optimizer = torch.optim.Adam(self.correction.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(self.correction.parameters(), lr=3e-4, weight_decay=0.01)
        nn.utils.clip_grad_norm_(self.correction.parameters(), 1.0)
                
        start = time.time()
        
        epoch_losses = []
                
        for e in range(epochs):
            epoch_loss = 0.0
            self.correction.train()
            
            for batch in dataloader:
                x = batch["x"]
                y_true = batch["y"]
                y_mask = batch.get("y_mask", None)
                
                if use_weather:
                    weather_cat = torch.cat([batch["x_weather"], batch["y_weather"]], dim=1)
                    weather = [weather_cat, batch["x_weather"], batch["y_weather"]]
                else:
                    weather = None
                if use_calendar:
                    calendar_cat = torch.cat([batch["x_calendar"], batch["y_calendar"]], dim=1)
                    calendar = [calendar_cat, batch["x_calendar"], batch["y_calendar"]]
                else:
                    calendar = None
                if use_latents:
                    use_latents = True
                else:
                    use_latents = None

                with torch.no_grad():
                    preds, latents = self.foundation.predict(
                        x, self.horizon, return_latents=True
                    )

                out = self.correction(preds, use_latents, weather, calendar)
                
                # out = preds + correction
                
                corr_res = y_true - out
                base_res = y_true - preds
                
                #loss = nn.MSELoss()(out, y_true)
                
                loss_zero = F.l1_loss(out, y_true)          # or MSE
                loss_hinge = F.relu((y_true - out).abs() - (y_true - preds).abs()).mean()
                loss = loss_zero + 0.5 * loss_hinge          # your λ
                
                # loss using the y_mask
                if y_mask is not None:
                    loss_zero = (y_mask * (out - y_true).abs()).sum() / (y_mask.sum() + 1e-6)
                    loss_hinge = (y_mask * F.relu(corr_res - base_res)).sum() / (y_mask.sum() + 1e-6)
                    loss = loss_zero + 0.5 * loss_hinge

                #loss = lambda_zero * loss_zero + lambda_hinge * loss_hinge
                
                print(f"Loss components ({loss.item():.4f}) - Zero: {loss_zero.item():.4f}, Hinge: {loss_hinge.item():.4f}", end="\r", flush=True)
                
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_losses.append(epoch_loss / len(dataloader))
            print(f"Epoch {e+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}", end="\r", flush=True)
        
        self.train_time = time.time() - start
        
        plt.figure()
        plt.plot(epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title(f"Error Correction Training Loss {self.mode}")
        # safe fig
        if model_name is not None:
            plt.savefig(f"results_server/loss_function_{model_name}.png")
        else:
            plt.show()
        plt.close()
    
    @torch.no_grad()
    def evaluate(self, dataloader, use_weather=True, use_calendar=True, use_latents=False, no_corrections=False):
        if self.correction is not None and not no_corrections:
            self.correction.eval()
        
        start = time.time()
        
        y_true, y_pred = [], []

        for batch in dataloader:
            x = batch["x"]
            y = batch["y"]
            
            if use_weather:
                weather = [batch["x_weather"], batch["y_weather"]]
            else:
                weather = None
            if use_calendar:
                calendar = [batch["x_calendar"], batch["y_calendar"]]
            else:
                calendar = None
            if use_latents:
                latents = True
            else:
                latents = None
            
            preds = self(x, latents=latents, weather=weather, calendar=calendar, no_corrections=no_corrections)
            y_true.append(y)
            y_pred.append(preds)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        
        self.eval_time = time.time() - start

        #print(f"Evaluation: MSE - {nn.MSELoss()(y_pred, y_true).item():.4f}, MAE - {nn.L1Loss()(y_pred, y_true).item():.4f} , train time: {self.train_time:.2f}s, eval time: {self.eval_time:.2f}s")
        
        return y_true, y_pred
