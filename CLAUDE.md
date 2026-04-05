# FM_cov_order — Project Knowledge for Claude

## Research Goal
PhD Paper 2. Ablation study on **Foundation-Model-based spatial-temporal forecasting**:
1. Which covariates help (weather, calendar, events)?
2. Past-only vs future-only covariate scope?
3. Does the **order** of processing modules (past covariates → spatial graph → future covariates, and permutations) affect accuracy?

---

## Dataset
| Setting | Modes | Path | Period |
|---------|-------|------|--------|
| Zurich (primary) | `road`, `bike`, `ped` | `data/train_test/` | 2022-2024, hourly |
| NYC benchmark | `nyc_bike`, `nyc_taxi`, `chi_taxi` | `data/train_test/benchmark/` | Apr–Jun 2016/2024 |

- Rolling windows: **input = 96 h (4 days), output = 18 h**
- Train/test split: **85/15 by time**
- Standard normalization: `rowwise_minmax` (per-sensor min-max)
- De-normalisation keys: `norm_params['X_max']`, `norm_params['X_min']`

**Covariate shapes per batch:**

| Key | Shape | Content |
|-----|-------|---------|
| `x_calendar` / `y_calendar` | `(B, T/H, 3)` | hour, day-of-week, month |
| `x_weather` / `y_weather` | `(B, T/H, 8)` | 8 reduced Swiss meteo features |
| `x_events` / `y_events` | `(B, T/H, K)` | binary holiday/event flags |

**GOTCHA — events were missing from `TimeSeriesDataset.__getitem__`**: `self.events_X/Y` were loaded but not returned in the dict, so events were silently ignored in all training runs. Fixed by adding `"x_events"` and `"y_events"` keys to `__getitem__` (same pattern as calendar/weather).

**GOTCHA — raw_fm MODULE_CONFIGS was always empty**: The filter `[c for c in MODULE_ORDER_CONFIGS if c["name"] == "raw_fm"]` returned `[]` since no entry is named `"raw_fm"`. Fixed to use `"graph_only"` as sentinel (module_order is irrelevant for raw_fm since `forward()` returns early before any module ordering). Event key checks in `train_model`/`evaluate_model` also updated to use `batch.get("x_events") is not None` instead of `"x_events" in batch` to handle None values correctly.

---

## CRITICAL GOTCHA — "MSE" column is actually RMSE
```python
# main.py line ~534 — sqrt IS applied:
mse_masked = np.sqrt(np.sum(((y_true - y_pred)**2) * y_mask) / np.sum(y_mask))
```
The result key `"MSE_masked"` and the `.tex` table column `MSE` store **RMSE values** (not MSE).
All existing numbers (e.g. 56.46 in the ped table) are RMSE.
**Any new metric code must apply `np.sqrt(...)` to stay comparable.**

---

## Core Model: `SpatialTemporalFM` (`models/FM_spatial_covariates.py`)
- Frozen FM backbone; each FM has a different hidden dim (see table below)
- FM adapter: `(B, D_fm, S) → (B, S, hidden_dim=256)` — `fm_hidden_dim` must match the FM
- Cross-attention fusion for past/future covariates (date, event, weather separately)
- 2 spatial graph attention layers (fully learned, no adjacency matrix by default)
- Final prediction = adapter output + FM residual
- `module_order` controls sequence: `"past_graph_future"`, `"graph_past_future"`, `"past_future_graph"`, `"graph_only"`, `"cov_only"`

**`cov_only` module order** (added 2026-04-01): Past Cov → Future Cov, **no spatial graph**. Ablation baseline to isolate the covariate contribution without spatial propagation. Graph layers are NOT built for `cov_only` (saves parameters; `_apply_graph_layers` has a `hasattr` guard and is a no-op if called). `cov_only` is **excluded from the ordering sweep** in `main.py` (`_skip = {"graph_only", "raw_fm", "cov_only"}`); run it via `SINGLE_CONFIG` or a dedicated loop.

**Selective module initialization**: `__init__` accepts all covariate flags (`use_date`, `use_event`, `use_weather`, `use_past_*`, `use_future_*`) plus `raw_fm=False`. It resolves these at construction time and only builds the encoder/cross-attention modules that will actually be used. `graph_only` forces all covariate flags to False. `cov_only` skips graph layer construction. `raw_fm=True` skips ALL module creation (adapter, graph layers, pred head, covariates) → 0 trainable parameters; `forward()` returns `fm_preds` immediately after `fm_encoder.predict()`. A `_device_ref` buffer is always registered so `.to(device)` and device queries work even with zero parameters. `_apply_past/future_covariates` use `hasattr` guards so forward() is safe even if a module was not built.

**FM wrappers** (all in `models/FM_spatial_covariates.py`) — must be instantiated BEFORE passing to `SpatialTemporalFM`:

| FM name | Class | Constructor args | `fm_hidden_dim` |
|---------|-------|-----------------|----------------|
| `"chronos"` | `Chronos2` | `()` | 6144 |
| `"moment"` | `MomentWrapper` | `(input_len, output_len)` | 1024 |
| `"timesfm25"` | `TimesFM25Wrapper` | `(device)` | 1280 |
| `"moirai2"` | `MOIRAI2Wrapper` | `(device)` | **384** (moirai-2.0-R-small, d_model=384) |
| `"flowstate"` | `FlowStateWrapper` | `(device)` | **512** (encoder_state_dim=512; `backbone_hidden_state[0]`) |

Each wrapper self-freezes its backbone. `SpatialTemporalFM` does NOT freeze the FM (fm_encoder is not an nn.Module submodule; its params are invisible to `model.parameters()`).

The `FM_HIDDEN_DIM` dict in `main.py` maps fm_name → correct dim and must be passed as `fm_hidden_dim=FM_HIDDEN_DIM[fm_name]` to `SpatialTemporalFM`.

**GOTCHA — FlowStateWrapper D was 256 (wrong), corrected to 512**: `granite-tsfm==0.3.5` / `tsfm_public` — `FlowStateForPrediction` outputs `backbone_hidden_state` with shape `(1, B*S, 512)` (encoder_state_dim=512). The old code used D=256 (decoder_dim), causing a shape mismatch in the fm_adapter. Fixed: `FlowStateWrapper.D = 512`, `FM_HIDDEN_DIM["flowstate"] = 512`. The hook was also removed — `backbone_hidden_state[0]` is now read directly from the `predict()` output, cached in `self._cache`, and consumed by `encode_hidden()`. The `prediction_outputs` tensor has shape `(B*S, H, 1)`; squeeze(-1) → `(B*S, H)`.

**GOTCHA — MOIRAI2Wrapper loads moirai-2.0-R-small manually**: `uni2ts==2.0.0`'s `MoiraiModule.from_pretrained()` fails for moirai-2.0 because the HF `config.json` uses `patch_size` (singular) but the class constructor requires `patch_sizes` (plural). Fix: manually read the config and construct `MoiraiModule(distr_output=StudentTOutput(), patch_sizes=(cfg["patch_size"],), ...)`, then `load_state_dict(load_file(...))`. As of 2026-04, only `moirai-2.0-R-small` is publicly available (base/large are gated/unavailable). Default `size="small"`, `D=384`, `patch_size=16`. MoiraiForecast output shape is `(B*S, samples, H)` (not `(B*S, samples, H, 1)` as earlier commented).

**GOTCHA — NaN values for bike/sparse data (fixed 2026-04-01)**: Two compounding bugs:
1. `data_loader.py apply_rowwise_minmax_normalization` had no epsilon guard on `(X_max - X_min)` denominator. Constant-value sensors (all-zero bike counters) produce `0/0 = NaN`. Fixed: `.clamp(min=1e-8)` on all three normalization denominators (lines 1012, 1028-1032). Same fix applied to `apply_minmax_normalization`.
2. NaN-valued series fed to Chronos2/any FM produce all-masked attention (`context_mask = ~isnan = all False`), which causes `softmax(-inf) = NaN` in SDPA. NaN from even ONE sensor then contaminates ALL sensors via spatial graph attention (Q@K^T NaN → whole matrix). Fix 2: `torch.nan_to_num(z, nan=0.0)` in `Chronos2.encode_hidden()` after pooling.

---

## Reference Models (`models/reference_models.py`)
Four non-FM baselines to challenge FM_order:

| Flag name | Class | Covariates | Architecture |
|-----------|-------|------------|--------------|
| `staef` | `STAEformer` | No | Adaptive per-sensor embedding as key/value in spatial attention (STAEformer style) |
| `stgformer` | `STGformer` | No | Factored linear spatial+temporal attention with recursive fusion |
| `exost` | `ExoST` | Yes (date+weather+events) | Chen et al. 2025 select-then-balance; Siamese ST encoders; context-aware balancer |
| `tftexost` | `TFTExoST` | Yes (date+weather+events) | LSTM enc/dec + temporal MHSA + ExoST spatial attention; additive covariate fusion |

All models: input `(B, T, S)` → output `(B, H, S)`.
`model.use_covariates` bool controls whether train/eval loops pass cov kwargs.
Results saved as `ref_{model_name}_{mode}` in `results_server/results.json`.

**Covariate key mapping in train/eval loops:**
- `ExoST`: receives `x_calendar`, `y_calendar`, `x_weather`, `y_weather`, `x_events`, `y_events` directly
- `CovTFT` / `TFTExoST`: loops remap `x_calendar→date_X`, `y_calendar→date_Y`, `x_weather→weather_X`, `y_weather→weather_Y`, `x_events→event_X`, `y_events→event_Y`

**Event dim detection in `__main__`:**
```python
e_d = tr_data.events_X.shape[2] if hasattr(tr_data, "events_X") else 0
```
Passed as `event_dim=e_d` to `ExoST`, `TFTExoST`, and `CovTFT` constructors.

**`run_reference_models()` — top-level callable (importable from `main.py`):**
```python
run_reference_models(modes, path, reference_model="all", epochs=100,
                     batch_size=32, lr=3e-4, dropout=0.1,
                     multi_run=False, n_runs=3, seeds=None)
```
Called from `main.py` at the end of the FM ablation loop (controlled by `RUN_REFERENCE_MODELS` flag). The `__main__` guard in `reference_models.py` still calls it directly so the file remains standalone-runnable.

**Train/eval functions:**
- `train_reference_model(model, loader, epochs, lr, name)`
- `evaluate_reference_model(model, loader, norm_params)`

**GOTCHA — `_IMHA` shared-V bug fixed:** `Wv` was `Linear(D, D)` but used with `view(B, L, 1, dh)` which required only `dh` elements. Fixed: `Wv = Linear(D, D//H)` (single head projection), then `unsqueeze(1).expand` to broadcast across all heads.

**GOTCHA — `TFTExoST` used undefined `GRN`:** All `GRN(hidden_dim, dropout=...)` calls replaced with `_GRN(hidden_dim, hidden_dim, dropout=...)` which is the correct local class.

---

## Results Infrastructure
```
results_server/results.json                          # shared dict, key = model_name
results_server/predictions/{model_name}_predictions.npz   # Y_pred, Y_true, Y_mask
results_server/loss_functions/loss_{model_name}.png  # training loss curve
```

**Result entry fields (single-run):** `model_name`, `mode`, `covariates_used`, `metrics`
(`MAE_masked`, `MSE_masked`[=RMSE!], `MAE_overall`, `MSE_overall`),
`train_time`, `inference_time`, `loss_history`, `number_of_parameters`, `hyperparameters`,
`n_runs` (=1), `seeds` (=[None]).

**Multi-run additions** (when `MULTI_RUN=True`, `n_runs > 1`):
- `metrics` = **mean** across runs (backward compatible — `result_analysis.py` unchanged)
- `metrics_std` = std across runs (same keys as `metrics`)
- `runs` = list of per-run dicts: `{"seed", "metrics", "train_time", "inference_time"}`
- `train_time` = sum across runs; `loss_history` = last run; predictions saved from last run
- `raw_fm` entries are **always single-run** (deterministic, no training)

---

## `main.py` Control Flags
All flags live near the top of the `if __name__ == '__main__':` block (around line 367):

| Flag | Default | Purpose |
|------|---------|---------|
| `foundation_spatial_covariates` | `False` | Gate for FM ablation loop |
| `MULTI_RUN` | `False` | Enable multi-seed runs |
| `N_RUNS` | `3` | Number of runs per config (when `MULTI_RUN=True`) |
| `BASE_SEEDS` | `[42, 123, 456]` | Seeds for each run |
| `RUN_REFERENCE_MODELS` | `True` | Run reference baselines after FM loop |
| `REFERENCE_MODEL` | `"all"` | Which reference model(s) to run |
| `REF_EPOCHS` | `300` | Epochs for reference models |
| `FOUNDATION_MODELS` | `["chronos","timesfm25","moment"]` | FMs to sweep |

**`set_seed(seed)`** helper in `main.py` sets Python/NumPy/PyTorch seeds for reproducibility.

---

## Training Details
**LR scheduler in `train_model()`** (`models/FM_spatial_covariates.py`):
`CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)` — decays from `lr` to `lr×0.01` over the full run. For `lr=3e-4`, `epochs=300`: 3e-4 → 3e-6. Current LR printed each epoch alongside loss.

---

## Result Analysis (`src/result_analysis.py`)

**`get_results(results_file, save_dir, filter_date)`** — main entry point; applies date filter, generates all plots and LaTeX tables.

**Date filtering** is applied upstream and passed as a `results` dict to sub-functions — NOT re-read from file — so the filter is respected everywhere.

**Key output functions:**

| Function | Output | Notes |
|----------|--------|-------|
| `generate_fm_text_summaries(...)` | `text_summary_fm_{fm}_{mode}.tex` | One table per FM×mode; rows = order+cov configs |
| `generate_mode_wide_tables(...)` | `wide_table_{mode}.tex` | One table per mode; rows = configs, cols = FM×{MAE,RMSE,Impr} |
| `generate_benchmark_table(...)` | `benchmark_table.tex` | Ref models vs TimesFM variants; bold=best, underline=2nd best |
| `generate_benchmark_timing_plots(...)` | `benchmark_timing_{train|infer}_{mode}.pdf` | 6 plots (3 modes × 2 time axes): MAE vs train/inference time; ref baselines + TimesFM raw/graph/cov/best |

All functions accept `results=<dict>` (pre-filtered) **and** `filter_date=<str>` for standalone calls.

**`generate_benchmark_timing_plots` model set per plot:**
- STAEformer, STGformer, ExoST, TFTExoST (from `all_unfiltered`, no date filter)
- TimesFM (raw) — hollow marker, key: `SpatialFM_order_{mode}_timesfm25_raw_fm_raw_fm`
- TimesFM + graph — filled light, key: `..._graph_only_no_cov`
- TimesFM + cov — filled, key: `..._cov_only_cov_only` (absent for `road`)
- TimesFM + best — larger marker, best MAE among graph+cov configs

**`wide_table` format:** `\tiny`, `table*`, moirai2 excluded, 4 FMs × 3 cols each. Improvement = % vs `raw_fm` baseline for that FM. Config rows sorted: raw_fm → graph_only → cov_only → gpf → pfg → pgf, then by covariate combo. `ORDER_TYPES` in all three parsing sites includes `"cov_only"`. `reshape_tex_tables` includes a `co` column (x = cov_only order). `ORDER_RANK`: raw_fm=0, graph_only=1, cov_only=2, graph_past_future=3, past_future_graph=4, past_graph_future=5.

**`wide_table` column flags (new format, 2026-04):** 4 flag columns replace the old 8: `fm` (always x), `g` (graph used), `cov` (covariates used), `order` (pgf/gpf/pfg when both g+cov active, blank otherwise). `wea`, `evt`, `date` columns were removed. `show_std=True` renders MAE/RMSE as `13.56 $\pm$ 0.50` (not inside `$...$`) and switches column alignment to `l l r` per FM group (MAE/RMSE left-aligned for readability, Impr right-aligned).

**`generate_benchmark_table` caption:** `best_configs_note` underscores replaced with spaces (`.replace('_', ' ')`) so LaTeX renders cleanly without `\_` escaping.

**GOTCHA — model name parsing in `generate_fm_text_summaries` / `generate_mode_wide_tables`:**
Old entries (pre-2026-03-20) use format `SpatialFM_order_{mode}_{order_type}_{exo}` (no FM name). New entries use `SpatialFM_order_{mode}_{fm}_{order_type}_{exo}`. Use `filter_date="2026-03-20"` to exclude old entries. ORDER_TYPES must be checked **before** the raw_fm fallback, otherwise entries like `{mode}_{fm}_{order_type}_raw_fm` are mis-parsed (order_type ends up in the fm field).

---

## Key File Map
| File | Purpose |
|------|---------|
| `main.py` | Training orchestration, ablation loops, `prepare_data()`, `set_seed()`, `save_results_to_json()` |
| `models/FM_spatial_covariates.py` | `SpatialTemporalFM`, FM wrappers, `train_model()` (with cosine LR), `evaluate_model()` |
| `models/reference_models.py` | `STGFormer`, `STAEformer`, `TFTExoST`, train/eval functions, `run_reference_models()` |
| `models/covariate_configs.py` | `PAST_FUTURE_CONFIGS`, `MODULE_ORDER_CONFIGS` |
| `src/data_loader.py` | `TimeSeriesDataset`, `SplitTimeSeriesSingleMode`, `covariateDataLoader` |
| `src/result_analysis.py` | `get_results()`, `generate_fm_text_summaries()`, `generate_mode_wide_tables()` |
