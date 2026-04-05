import os
import shutil
import pandas as pd
import numpy as np
from src.data_loader import TrafficDataManager, ActiveModeDataLoader, TransitDataLoader, SplitTimeSeriesSingleMode, covariateDataLoader
#from models.train_models import train_model_mode, train_model_nw
#from models.train_models import evaluate_models
#from src.data_descriptive import check_correlation, check_temporal_correlation, check_multistep_correlation
#from src import data_descriptive
#from models.d_model_clustering.cluster import cluster_data, plot_clusters
import src.data_loader as dl
from src.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader, random_split


#from src.result_visualization import plot_comparisons, plot_all_results
#from src.train_fl import fl_training
#from src.spatial_data_loader import get_all_sensor_locations, create_adj_matrix
from src.data_descriptive import plot_sensor_availability, drop_useless_sensors
import torch

#from models.FED2 import FED2, temporal_model
# Legacy FM wrappers (kept for reference; active wrappers are in models.FM_spatial_covariates)
# from models.foundation_models import Chronos2Wrapper, TimesFMWrapper, TTMWrapper, ExtendedFM, ErrorCorrectionModule, Chronos2WrapperConditional


#import src.result_visualization as vis

from datetime import datetime
import json
import shutil
import tempfile
import time
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def collate_with_optional(batch):
    collated = {}
    keys = batch[0].keys()
    
    for k in keys:
        vals = [d[k] for d in batch]
        if vals[0] is None:
            collated[k] = None
        else:
            collated[k] = torch.stack(vals, dim=0)
    
    return collated


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_covariate_flags(config):
    """Extract all covariate flags from config dict"""
    flags = {
        'use_date': False,
        'use_event': False,
        'use_weather': False,
        'use_past_date': None,
        'use_past_event': None,
        'use_past_weather': None,
        'use_future_date': None,
        'use_future_event': None,
        'use_future_weather': None,
        'raw_FM': False
    }
    flags.update({k: v for k, v in config.items() if k in flags})
    return flags


def save_results_to_json(model_name, results_dict, results_file="results_server/results.json"):
    """Save results to JSON file"""
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = {}
    else:
        existing_results = {}
    
    # Handle old list format
    if isinstance(existing_results, list):
        tmp = {}
        for r in existing_results:
            if "model_name" in r:
                tmp[r["model_name"]] = r
        existing_results = tmp
    
    existing_results[model_name] = results_dict
    
    try:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp_file:
            json.dump(existing_results, tmp_file, indent=2)
            temp_name = tmp_file.name
        shutil.move(temp_name, results_file)
    except Exception as e:
        print(f"Failed to save results: {e}")


def print_paradigm_summary(paradigm_results, mode, paradigm):
    """Print summary table for a paradigm"""
    if not paradigm_results:
        return
    
    print(f"\n{'='*100}")
    print(f"SUMMARY - {mode.upper()} - {paradigm.upper()} Paradigm")
    print(f"{'='*100}")
    print(f"{'Config':<25} {'MAE':<12} {'MSE':<12} {'Time (s)':<10}")
    print(f"{'-'*100}")
    
    sorted_results = sorted(paradigm_results.items(), 
                          key=lambda x: x[1]["metrics"]["MAE_masked"])
    
    for config_name, res in sorted_results:
        mae = res["metrics"]["MAE_masked"]
        mse = res["metrics"]["MSE_masked"]
        train_t = res["train_time"]
        print(f"{config_name:<25} {mae:<12.4f} {mse:<12.4f} {train_t:<10.2f}")
    
    best = sorted_results[0]
    print(f"{'-'*100}")
    print(f"Best: {best[0]} (MAE: {best[1]['metrics']['MAE_masked']:.4f})")
    print(f"{'='*100}\n")
    
def set_seed(seed: int):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(mode, path, normalization_type, batch_size, exogen_var=False, exo_reduced=True, add_input_feature_dim=False):
    # Train data: fit and store normalization params
    train_data = TimeSeriesDataset(mode=mode, path=path, train=True, exogen_var=exogen_var, exo_reduced=exo_reduced, normalization_type=normalization_type)
    # Test data: use train's params for normalization
    test_data = TimeSeriesDataset(
        mode=mode, path=path, train=False, exogen_var=exogen_var, exo_reduced=exo_reduced, normalization_type=normalization_type, norm_params=train_data.norm_params)
    
    if add_input_feature_dim:
        train_data.add_input_feature_dim()
        test_data.add_input_feature_dim()
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_with_optional)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_with_optional)
    #return train_data, test_data, train_loader, test_loader, similarity_set if similarity_set is not None else None, val_loader if validation else None
    return (
        train_data,
        test_data,
        train_loader,
        test_loader
    )

    
def save_predictions(path, y_pred, y_true, y_mask):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_mask = y_mask.cpu().numpy()
    
    np.savez(path, Y_pred=y_pred, Y_true=y_true, Y_mask=y_mask)



if __name__ == '__main__':
    start_month = "2022-01"
    end_month = "2025-12"
    
    start_month_nyc = "2016-04"
    end_month_nyc = "2016-06"
    start_month_chi = "2024-04"
    end_month_chi = "2024-06"
    
    split_ratio = 0.75
    split_date = "2025-01-01"
    input_size = 4*24
    output_size = 18
    
    LOAD_DATA = False 
    
    # Print their shapes
    road_train = np.load("data/train_test/road_train_X.npy")
    road_test = np.load("data/train_test/road_test_X.npy")
    
    print(f"Shape of road_train_X: {road_train.shape}")
    print(f"Shape of road_test_X: {road_test.shape}")
    
    
    if LOAD_DATA:
        # Load and preprocess weather data
        covariateDL = covariateDataLoader(data_path="data/raw/", start_month=start_month, end_month=end_month)
        
        weather_TS = SplitTimeSeriesSingleMode(covariateDL.data_weather, start_month, end_month, mode="weather", transpose=True)
        weather_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        weather_TS.store_data()
        
        weather_DL_short = pd.DataFrame(index=covariateDL.data_weather.index)
        # group get mean for each of these strings of coulumn that contain it ["fkl010h0", "fkl010h3", "htoauths", "rre150h0", "sre000h0", "ure200h0","tre200h0","tre005h0"
        keys = [
            "fkl010h0", "fkl010h3", "htoauths", "rre150h0",
            "sre000h0", "ure200h0", "tre200h0", "tre005h0"
        ]
        for k in keys:
            cols = [c for c in covariateDL.data_weather.columns if k in c]
            weather_DL_short[k] = covariateDL.data_weather[cols].mean(axis=1, skipna=True)
        weather_DL_short.to_csv("data/processed/weather_zh_reduced.csv", index=True)
        weather_TS_short = SplitTimeSeriesSingleMode(weather_DL_short, start_month, end_month, mode="weather_reduced", transpose=True)
        weather_TS_short.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        weather_TS_short.store_data()
        
        events_TS = SplitTimeSeriesSingleMode(covariateDL.data_events, start_month, end_month, mode="events", transpose=True)
        events_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        events_TS.store_data()
        
        calendar_TS = SplitTimeSeriesSingleMode(covariateDL.data_calendar, start_month, end_month, mode="calendar", transpose=True)
        calendar_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        calendar_TS.store_data()
        
        
        """
        weather_TS.load_data()
        weather_TS.preprocess_holiday_data()
        
        weather_TS.data.to_csv("data/processed/external_factors_raw_zh.csv", index=True)
        
        # normalize min-max per column
        weather_TS.normalize_min_max()
        
        # store weather data as csv
        weather_TS.data.to_csv("data/processed/external_factors_zh.csv", index=True)
        
        # The df could be reduced to less features (deleting columns, PCA or embedding space)
        
        exo_TS = SplitTimeSeriesSingleMode(weather_TS.data, start_month, end_month, mode="exoVar", transpose=True)
        exo_TS.split_training_testing(ratio=0.85, input_size=4*24, output_size=18)
        exo_TS.store_data()
        """

    
    # check if there are npy file (bike_test_X, bike_test_Y, ped_train_X, ped_train_Y, road_test_X_mask, road_test_Y_mask) in data/train_test
    if LOAD_DATA:
    
        # Load and preprocess the data
        df_road, df_bike, df_pedestrian = dl.load_data()
        print("\nData loaded\n")
        
        # Check that both data have the same columns
        only_bike = df_bike.columns.difference(df_road.columns)
        df_bike = df_bike.drop(columns=only_bike)
        df_pedestrian = df_pedestrian.drop(columns=only_bike)
        
        # Plot the availability of the data
        #availability = plot_sensor_availability([df_road, df_bike, df_pedestrian], ["Road", "Bike", "Pedestrian"])
        #availability_22_24 = plot_sensor_availability([df_road, df_bike, df_pedestrian], ["Road", "Bike", "Pedestrian"], start=start_month, end=end_month)
        
        [df_road, df_bike, df_pedestrian], deleted_idx = drop_useless_sensors([df_road, df_bike, df_pedestrian], ["Road", "Bike", "Pedestrian"], start=start_month, end=end_month)
        df_road.to_csv("data/processed/MIV/traffic_all_years_filtered.csv", index=True)
        df_bike.to_csv("data/processed/Langsamverkehr/bike_all_years_filtered.csv", index=True)
        df_pedestrian.to_csv("data/processed/Langsamverkehr/pedestrian_all_years_filtered.csv", index=True)
        
        """
        
        #####################################################################################
        

        
        test_corr = False
        if test_corr:
            # Check correlation of the data
            check_correlation(data_road.T, "road")
            
            # Get the relative change from one column to the next for each individual sensor
            data_road_rel = data_road.T.pct_change().T
            # Drop nan column
            data_road_rel.dropna(axis=1, inplace=True)
            check_correlation(data_road_rel.T, "road_rel")
            
            check_multistep_correlation(data_road, "road")
            check_temporal_correlation(data_road, "road")
        
        vbz = False
        if vbz:
            # Load raw VBZ data
            folder_path = r"data/raw/VBZ/Rohdaten"
            # Get path of parqeuet files in the folder
            parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
            
            # read parquet files in the folder
            raw_VBZ = pd.read_parquet(parquet_files[0])
            print(raw_VBZ.head())
            print(raw_VBZ.columns)
            print(raw_VBZ.shape)
        
        """
        ################################################################################################
        # %% Split the data for single mode prediction
        
        shape_check = True
        if shape_check:
            print(f"Road: {df_road.shape}")
            print(f"Bike: {df_bike.shape}")
            print(f"Pedestrian: {df_pedestrian.shape}")
        
        bike_TS = SplitTimeSeriesSingleMode(df_bike, start_month, end_month, mode="bike")
        bike_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size) # Also possible with split date, little problem in the implementation
        bike_TS.store_data()
        
        pedestrian_TS = SplitTimeSeriesSingleMode(df_pedestrian, start_month, end_month, mode="ped")
        pedestrian_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        pedestrian_TS.store_data()
        
        road_TS = SplitTimeSeriesSingleMode(df_road, start_month, end_month, mode="road")
        road_TS.split_training_testing(split_date=split_date, input_size=input_size, output_size=output_size)
        road_TS.store_data()
        
        print("Data split and stored")

    if not os.path.exists("data/train_test/benchmark/nyc_bike_train_X.npy"):
        # if not exist, creat benchmark folder
        os.makedirs("data/train_test/benchmark", exist_ok=True)
        # Load and preprocess the benchmark data
        nyc_bike = pd.read_csv("data/processed/benchmark/nyc_bike_tripdata_full.csv", index_col=0, parse_dates=True)
        nyc_taxi = pd.read_csv("data/processed/benchmark/nyc_taxi_tripdata_full.csv", index_col=0, parse_dates=True)
        chi_taxi = pd.read_csv("data/processed/benchmark/chi_taxi_tripdata_full.csv", index_col=0, parse_dates=True)
        
    # SplitTimeSeriesSingleMode: transpose data
    # Split it
        nyc_bike_TS = SplitTimeSeriesSingleMode(nyc_bike.T, start_month_nyc, end_month_nyc, mode="nyc_bike")
        nyc_bike_TS.split_training_testing(ratio=0.85, input_size=12, output_size=1)
        nyc_bike_TS.store_data(suffix="benchmark")
        
        nyc_taxi_TS = SplitTimeSeriesSingleMode(nyc_taxi.T, start_month_nyc, end_month_nyc, mode="nyc_taxi")
        nyc_taxi_TS.split_training_testing(ratio=0.85, input_size=12, output_size=1)
        nyc_taxi_TS.store_data(suffix="benchmark")
        
        chi_taxi_TS = SplitTimeSeriesSingleMode(chi_taxi.T, start_month_chi, end_month_chi, mode="chi_taxi")
        chi_taxi_TS.split_training_testing(ratio=0.85, input_size=12, output_size=1)
        chi_taxi_TS.store_data(suffix="benchmark")

    ########################################################################################################################################################################
    
    data_setting = {
        "zurich": {"modes": ["ped", "bike", "road"],
                 "path": "data/train_test"
        },
        "nyc": {"modes": ["nyc_bike", "nyc_taxi", "chi_taxi"],
              "path": "data/train_test/benchmark"},
    }
    
    setting = "zurich"  # "zurich" or "nyc"
    modes = data_setting[setting]["modes"]
    path = data_setting[setting]["path"]
    
    ########################################################################################################################################################################
    
    
    batch_size = 32

    # ── Multi-run & reference model flags ─────────────────────────────────────
    MULTI_RUN            = False          # enable multiple runs with different seeds
    N_RUNS               = 3             # number of runs (used when MULTI_RUN=True)
    BASE_SEEDS           = [42, 123, 456] # seeds for each run
    RUN_REFERENCE_MODELS = False          # run reference baselines after FM loop
    REFERENCE_MODEL      = "all"         # "staef"|"stgformer"|"exost"|"tftexost"|"all"
    REF_EPOCHS           = 300
    epochs = 2

    foundation_spatial_covariates = False


    FM_MODEL = "moment"  # "chronos" or "moment" — MOMENT supports backprop through FM
    FOUNDATION_MODELS = ["chronos", "moment", "timesfm25"] # moirai2, flowstate

    # Hidden dim of each FM's encode_hidden() output
    # moirai2 uses moirai-2.0-R-small (d_model=384, only public size as of 2026-04)
    # flowstate: encoder_state_dim=512 (backbone_hidden_state, not decoder_dim=256)
    FM_HIDDEN_DIM = {
        "chronos":   6144,
        "moment":    1024,
        "timesfm25": 1280,
        "moirai2":    384,
        "flowstate":  512,
    }

    normalization_types = ["minmax", "rowwise_minmax"]  # Add more as needed
    #norm = "_rownorm" if normalization_type == "rowwise_minmax" else "_norm"
    norm_dict = {
        "rowwise_minmax": "rownorm",
        "minmax": "norm"
    }
    
    collaboration = [True, False]
    


    if foundation_spatial_covariates:
        # ========================================================================
        # CONFIGURATION OPTIONS
        # ========================================================================
        from models.covariate_configs import PAST_FUTURE_CONFIGS, MODULE_ORDER_CONFIGS
        from models.FM_spatial_covariates import SpatialTemporalFM, train_model, evaluate_model, Chronos2
        from models.FM_spatial_covariates import Chronos2, MomentWrapper, TimesFM25Wrapper, MOIRAI2Wrapper, FlowStateWrapper
        
        # Option 1: Test ALL configurations (full ablation study)
        RUN_ABLATION = False  # Set to False to test only single config
        TEST_PAST_FUTURE = False
        TEST_ORDER = True
        
        # Option 2: Test SINGLE configuration (when RUN_ABLATION=False)
        SINGLE_CONFIG = {
            "name": "all_cov",
            "use_date": True,
            "use_event": True,
            "use_weather": True
        }
        
        # Define all configurations for ablation study
        BASIC_CONFIGS = [
            {"name": "raw_fm", "raw_fm": True, "use_date": False, "use_event": False, "use_weather": False},
            {"name": "no_cov", "use_date": False, "use_event": False, "use_weather": False},
            {"name": "date", "use_date": True, "use_event": False, "use_weather": False},
            {"name": "weather", "use_date": False, "use_event": False, "use_weather": True},
            {"name": "event", "use_date": False, "use_event": True, "use_weather": False},
            {"name": "date_weather", "use_date": True, "use_event": False, "use_weather": True},
            {"name": "event_weather", "use_date": False, "use_event": True, "use_weather": True},
            {"name": "date_event", "use_date": True, "use_event": True, "use_weather": False},
            {"name": "all_cov", "use_date": True, "use_event": True, "use_weather": True},
        ]
        SINGLE_CONFIG = [{"name": "raw_fm", "raw_fm": True, "use_date": False, "use_event": False, "use_weather": False},
                         {"name": "no_cov", "use_date": False, "use_event": False, "use_weather": False},
                         {"name": "all_cov", "use_date": True, "use_event": True, "use_weather": True}]
        
        SINGLE_CONFIG = [
            {"name": "cov_only", "raw_fm": False, "use_date": True, "use_event": True, "use_weather": True}
        ]
        
        if RUN_ABLATION:
            if TEST_PAST_FUTURE:
                covariate_configs = PAST_FUTURE_CONFIGS
            else:
                covariate_configs = BASIC_CONFIGS
        else:
            covariate_configs = SINGLE_CONFIG
            
        
        # Select which configs to run
        #covariate_configs = ALL_CONFIGS if RUN_ABLATION else [SINGLE_CONFIG]
        
        
        def get_covariate_flags(config):
            flags = {
                "use_date": False,
                "use_event": False,
                "use_weather": False,
                "use_past_date": None,
                "use_future_date": None,
                "use_past_event": None,
                "use_future_event": None,
                "use_past_weather": None,
                "use_future_weather": None,
                "raw_fm": False
            }
            flags.update({k: v for k, v in config.items() if k in flags})
            return flags
        
        
        # ========================================================================
        # MAIN TRAINING LOOP
        # ========================================================================
        
        for mode in modes:
            # Configuration
            batch_size = 32
            hidden_dim = 256
            graph_layers = 2
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            train_data, test_data, train_loader, test_loader = prepare_data(
                mode, path, 
                exogen_var=True, 
                exo_reduced=True, 
                normalization_type="rowwise_minmax", 
                batch_size=batch_size
            )
        
            # Get dimensions
            input_len = train_data.input_dim
            output_len = train_data.output_dim
            num_sensors = train_data.num_sensors
            weather_dim = train_data.weather_X.shape[2]
            event_dim = train_data.events_X.shape[2]
            date_dim = train_data.calendar_X.shape[2]
        
        
            # Store results for comparison
            mode_results = {}
            
            # Loop through the foundation models
            for fm_name in FOUNDATION_MODELS:

                # Loop through configurations
                for config in covariate_configs:
                    config_name = config["name"]
                    flags = get_covariate_flags(config)
                    
                    # if raw_fm or no covariates, only test one ordering (chronos or moment)
                    if flags['raw_fm']:
                        # raw_fm ignores module_order (forward returns fm_preds immediately); use raw_fm as sentinel
                        MODULE_CONFIGS = [c for c in MODULE_ORDER_CONFIGS if c["name"] == "raw_fm"]
                    elif (not flags['use_date'] and not flags['use_event'] and not flags['use_weather']):
                        # No covariates: spatial-only baseline
                        MODULE_CONFIGS = [c for c in MODULE_ORDER_CONFIGS if c["name"] == "graph_only"]
                    elif config_name == "cov_only":
                        # cov_only covariate config: fixed module order, no graph — single run
                        MODULE_CONFIGS = [c for c in MODULE_ORDER_CONFIGS if c["name"] == "cov_only"]
                    else:
                        # Ordering sweep: pgf / gpf / pfg only (graph + covariates variants)
                        _skip = {"graph_only", "raw_fm", "cov_only"}
                        MODULE_CONFIGS = [c for c in MODULE_ORDER_CONFIGS if c["name"] not in _skip]
                    
                    for order_config in MODULE_CONFIGS:
                        module_order = order_config["name"]
                        model_name = f"SpatialFM_order_{mode}_{fm_name}_{module_order}_{config_name}"

                        print(f"Spatial FM - Mode: {mode} | FM: {fm_name} | Config: {config_name} | Order: {module_order}")

                        # raw_fm is deterministic (no training) — always single run
                        _n_runs = 1 if flags['raw_fm'] else (N_RUNS if MULTI_RUN else 1)
                        _seeds  = BASE_SEEDS[:_n_runs] if (MULTI_RUN and not flags['raw_fm']) else [None]
                        run_records = []

                        for seed in _seeds:
                            if seed is not None:
                                set_seed(seed)

                            # Instantiate FM wrapper fresh for each run
                            if fm_name == "chronos":
                                fm_instance = Chronos2()
                            elif fm_name == "moment":
                                fm_instance = MomentWrapper(input_len=input_len, output_len=output_len)
                            elif fm_name == "timesfm25":
                                fm_instance = TimesFM25Wrapper(device=device)
                            elif fm_name == "moirai2":
                                fm_instance = MOIRAI2Wrapper(device=device)
                            else:  # flowstate
                                fm_instance = FlowStateWrapper(device=device)
                            fm_hidden_dim = FM_HIDDEN_DIM[fm_name]

                            model = SpatialTemporalFM(
                                fm_encoder=fm_instance,
                                num_sensors=num_sensors,
                                input_len=input_len,
                                output_len=output_len,
                                date_dim=date_dim,
                                event_dim=event_dim,
                                weather_dim=weather_dim,
                                fm_hidden_dim=fm_hidden_dim,
                                hidden_dim=hidden_dim,
                                graph_layers=graph_layers,
                                use_road_network=False,
                                module_order=module_order,
                                use_date=flags['use_date'],
                                use_event=flags['use_event'],
                                use_weather=flags['use_weather'],
                                use_past_date=flags['use_past_date'],
                                use_past_event=flags['use_past_event'],
                                use_past_weather=flags['use_past_weather'],
                                use_future_date=flags['use_future_date'],
                                use_future_event=flags['use_future_event'],
                                use_future_weather=flags['use_future_weather'],
                                raw_fm=flags['raw_fm'],
                                device=device
                            ).to(device)

                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                            # Train
                            if not flags['raw_fm']:
                                train_time = train_model(
                                    model, train_loader,
                                    epochs=epochs,
                                    model_name=model_name,
                                    **flags
                                )
                            else:
                                train_time = 0
                                model.loss_history = [0]

                            # Evaluate
                            y_true, y_pred = evaluate_model(
                                model, test_loader,
                                test_data.norm_params,
                                **flags
                            )

                            # Metrics
                            y_mask = test_data.Y_mask.numpy()
                            mse_masked = np.sqrt(np.sum(((y_true - y_pred) ** 2) * y_mask) / np.sum(y_mask))
                            mae_masked = np.sum(np.abs(y_true - y_pred) * y_mask) / np.sum(y_mask)
                            mse_overall = np.sqrt(np.mean((y_true - y_pred) ** 2))
                            mae_overall = np.mean(np.abs(y_true - y_pred))

                            print(f"  seed={seed}  MAE={mae_masked:.4f}  RMSE={mse_masked:.4f}  train={train_time:.1f}s")
                            run_records.append({
                                "seed": seed,
                                "metrics": {
                                    "MAE_masked":  float(mae_masked),
                                    "MSE_masked":  float(mse_masked),
                                    "MAE_overall": float(mae_overall),
                                    "MSE_overall": float(mse_overall),
                                },
                                "train_time":     train_time,
                                "inference_time": model.eval_time,
                            })

                        # Save predictions from last run
                        save_path = f"results_server/predictions/{model_name}_predictions.npz"
                        np.savez(save_path, Y_pred=y_pred, Y_true=y_true, Y_mask=y_mask)

                        # Aggregate across runs
                        all_m = {k: [r["metrics"][k] for r in run_records]
                                 for k in run_records[0]["metrics"]}
                        metrics_mean = {k: float(np.mean(v)) for k, v in all_m.items()}
                        metrics_std  = {k: float(np.std(v))  for k, v in all_m.items()} \
                                       if _n_runs > 1 else None

                        result_entry = {
                            "model_name":       model_name,
                            "mode":             mode,
                            "covariate_config": config_name,
                            "covariates_used":  flags,
                            "creation_time":    time.strftime("%Y-%m-%d %H:%M:%S"),
                            "n_runs":           _n_runs,
                            "seeds":            _seeds,
                            "metrics":          metrics_mean,
                            "metrics_std":      metrics_std,
                            "runs":             run_records if _n_runs > 1 else None,
                            "train_time":       sum(r["train_time"] for r in run_records),
                            "inference_time":   run_records[-1]["inference_time"],
                            "loss_history":     [float(x) for x in model.loss_history],
                            "number_of_parameters": trainable_params,
                            "hyperparameters":  {
                                "hidden_dim":       hidden_dim,
                                "graph_layers":     graph_layers,
                                "use_road_network": False,
                                "epochs":           epochs,
                                "batch_size":       batch_size,
                            },
                        }
                        mode_results[config_name] = result_entry
                        save_results_to_json(model_name, result_entry)

    if RUN_REFERENCE_MODELS:
        from models.reference_models import run_reference_models
        run_reference_models(
            modes=modes,
            path=path,
            reference_model=REFERENCE_MODEL,
            epochs=REF_EPOCHS,
            batch_size=batch_size,
            multi_run=MULTI_RUN,
            n_runs=N_RUNS,
            seeds=BASE_SEEDS,
        )

    from src.result_analysis import get_results
    get_results(filter_date="2026-03-24")
    