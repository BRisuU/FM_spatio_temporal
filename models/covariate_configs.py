# Updated configurations for testing past vs future covariates separately

# Define all configurations including past/future separation tests
PAST_FUTURE_CONFIGS = [
    # ========================================================================
    # PAST vs FUTURE COVARIATE TESTS
    # ========================================================================
    
    # PAST ONLY tests
    {"name": "past_date_only", 
        "use_past_date": True, "use_past_event": False, "use_past_weather": False,
        "use_future_date": False, "use_future_event": False, "use_future_weather": False},
    
    {"name": "past_weather_only", 
        "use_past_date": False, "use_past_event": False, "use_past_weather": True,
        "use_future_date": False, "use_future_event": False, "use_future_weather": False},
    
    {"name": "past_event_only", 
        "use_past_date": False, "use_past_event": True, "use_past_weather": False,
        "use_future_date": False, "use_future_event": False, "use_future_weather": False},
    
    {"name": "past_all", 
        "use_past_date": True, "use_past_event": True, "use_past_weather": True,
        "use_future_date": False, "use_future_event": False, "use_future_weather": False},
        
    # FUTURE ONLY tests
    {"name": "future_date_only", 
        "use_past_date": False, "use_past_event": False, "use_past_weather": False,
        "use_future_date": True, "use_future_event": False, "use_future_weather": False},
    
    {"name": "future_weather_only", 
        "use_past_date": False, "use_past_event": False, "use_past_weather": False,
        "use_future_date": False, "use_future_event": False, "use_future_weather": True},
    
    {"name": "future_event_only", 
        "use_past_date": False, "use_past_event": False, "use_past_weather": False,
        "use_future_date": False, "use_future_event": True, "use_future_weather": False},
    
    {"name": "future_all", 
        "use_past_date": False, "use_past_event": False, "use_past_weather": False,
        "use_future_date": True, "use_future_event": True, "use_future_weather": True},
    
    # MIXED tests (e.g., past date + future weather)
    {"name": "past_date_future_weather", 
        "use_past_date": True, "use_past_event": False, "use_past_weather": False,
        "use_future_date": False, "use_future_event": False, "use_future_weather": True},
    
    {"name": "past_date_future_event", 
        "use_past_date": True, "use_past_event": False, "use_past_weather": False,
        "use_future_date": False, "use_future_event": True, "use_future_weather": False},
]


# Helper function to extract flags from config
def get_covariate_flags(config):
    """Extract all covariate flags from config dict"""
    # Default values
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
        'raw_fm': False
    }
    
    # Update with config values
    flags.update({k: v for k, v in config.items() if k in flags})
    
    return flags

# Define module order configurations to test
MODULE_ORDER_CONFIGS = [
    {
        "name": "past_graph_future",
        "order": "past_graph_future",
        "description": "Past Cov → Graph → Future Cov (CURRENT)",
        "rationale": "Condition on history, aggregate spatially, refine with future"
    },
    {
        "name": "graph_past_future",
        "order": "graph_past_future",
        "description": "Graph → Past Cov → Future Cov",
        "rationale": "Spatial aggregation first, then temporal conditioning"
    },
    {
        "name": "past_future_graph",
        "order": "past_future_graph",
        "description": "Past Cov → Future Cov → Graph",
        "rationale": "All temporal conditioning first, then spatial propagation"
    },
    {
        "name": "graph_only",
        "order": "graph_only",
        "description": "Graph only (no covariates)",
        "rationale": "Baseline: spatial aggregation without temporal conditioning"
    },
    {
        "name": "cov_only",
        "order": "cov_only",
        "description": "Covariates only: Past Cov → Future Cov (no spatial graph)",
        "rationale": "Ablation: isolate covariate contribution without spatial propagation"
    },
    {
        "name": "raw_fm",
        "order": "raw_fm",
        "description": "Raw FM (no covariates, no graph)",
        "rationale": "Baseline: no spatial or temporal conditioning"
    }
]


# Example usage in training loop:
"""
for config in ALL_CONFIGS:
    config_name = config["name"]
    flags = get_covariate_flags(config)
    
    # Train
    train_time = train_model(
        model, train_loader, 
        epochs=epochs, 
        model_name=f"SpatialFM_{mode}_{config_name}",
        **flags  # Unpack all covariate flags
    )
    
    # Evaluate
    y_true, y_pred = evaluate_model(
        model, test_loader, 
        test_data.norm_params,
        **flags  # Unpack all covariate flags
    )
"""