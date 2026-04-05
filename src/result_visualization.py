import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

import numpy as np
import torch
import math
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages


def plot_all_results(model_name, results_file):

    
    df = pd.read_json(results_file)
    df = df.T
    
    df = df.drop(columns='metrics').join(df['metrics'].apply(pd.Series))
    
    # Now colum taking last after the last _ in column model_name
    df['mode'] = df['model_name'].apply(lambda x: x.split('_')[-1])
    
    # Only keep columns model_name, mode, MAE, RMSE
    df = df[['model_name', 'mode', 'MAE', 'RMSE']]
    print(df.columns)

    
    pivot = df.pivot(index='model_name', columns='mode', values=['MAE', 'RMSE'])
    
    # Remove the last part from the model name after the last _
    pivot.index = pivot.index.str.rsplit('_', n=1).str[0]
    # replace nan with 0
    pivot = pivot.fillna(0)
    
    # group by model name and take the max
    pivot = pivot.groupby(pivot.index).max()
    
    order = ["FedSTD", "STD", "FedSD", "SD", "FedTD", "TD"]
    def model_order(name):
        base = next((i for i, k in enumerate(order) if name.startswith(k)), len(order))
        exo_bonus = 0 if '_exo' in name else 1  # exo first
        return base * 2 + exo_bonus

    pivot = pivot.loc[sorted(pivot.index, key=model_order)]
    pivot = pivot.swaplevel(axis=1).sort_index(axis=1, level=0)
    

    
    
    print(pivot.head(15))
    
    latex_str = pivot.to_latex(
        multirow=True,
        multicolumn=True,
        float_format="%.2f",
        caption="Model performance per mode (MAE and RMSE)",
        label="tab:model_results"
    )

    print(latex_str)


    
    lal
    
    # make a table with the 
    # model name, mode0, mode1, mode2
    # STD (MAE, RMSE)...
    # FEDSTD



def plot_comparisons(mode=None, model_names=None, results_file="results/results.json", samples=None):
    """
    Plot runtime comparisons and sample predictions for selected models.
    :param model_names: List of model names to compare.
    :param results_file: Path to the centralized results file.
    :param samples: Optional list of sample indices for prediction comparison.
    """
    # Get the time in yymmddhhmmss format
    now = datetime.now()
    now = now.strftime("%y%m%d_%H%M%S")
    
    if mode == None:
        mode = "all"
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return
    
    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Get the model names from the results file
    if model_names is None:
        model_names = [v['model_name'] for v in all_results.values()]
        # Only keep the model names that contain the mode
        model_names = [name for name in model_names if mode in name]
        print(f"Model names: {model_names}")
    
    train_times = []
    inference_times = []
    loss_histories = {}

    for model_name in model_names:
        if model_name in all_results:
            train_times.append(all_results[model_name]["train_time"])
            inference_times.append(all_results[model_name]["inference_time"])
            loss_histories[model_name] = all_results[model_name]["loss_history"]
            
    # MAke a table with model names, training times, trainingtimes per epochs (training time / length of loss history) inference times, and number of parameter
    table = []
    for model_name in model_names:
        if model_name in all_results:
            model_results = all_results[model_name]
            train_time = model_results["train_time"]
            inference_time = model_results["inference_time"]
            num_parameters = model_results.get("number_of_parameters", "N/A")
            epochs = len(model_results["loss_history"])
            train_time_per_epoch = train_time / epochs if epochs > 0 else "N/A"
            
            table.append({
                "Model Name": model_name,
                "Training Time (s)": train_time,
                "Training Time per Epoch (s)": train_time_per_epoch,
                "Inference Time (s)": inference_time,
                "Number of Parameters": num_parameters
            })
    print("Results Table:")
    time_table = pd.DataFrame(table)
    time_table = time_table.round(2)  # Round to 2 decimal places
    time_table.set_index("Model Name", inplace=True)
    print(time_table)
    # Safe as latex table
    time_table.to_latex(
        f"results/tables/{now}_{mode}_time_comparison.tex",
        index=True,
        float_format="%.2f",
        escape=False,
        caption=f"Runtime Comparison for {mode} mode",
        label=f"tab:runtime_comparison_{mode}",
        column_format="|l|" + "l|" * len(time_table.columns),
        #hrules=True  # Only works in pandas >= 2.0,
    )

    
    # Plot runtime comparisons
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(model_names, train_times, color="blue", alpha=0.7)
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    # rotate x labels for better visibility
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    plt.bar(model_names, inference_times, color="green", alpha=0.7)
    plt.title("Inference Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.savefig(f"results/plots/time_{mode}_{now}.png")
    plt.xticks(rotation=45, ha='right')
    
    # Plot loss curves
    # Start a new plot for loss development
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        if model_name in loss_histories:
            plt.plot(loss_histories[model_name], label=model_name)
            
    # Set x lim to 250
    plt.xlim(0, 250)
    
    plt.title("Loss Development")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    #plt.savefig(f"results/plots/loss_development_{mode}_{now}.png")
    
def format_model_name(name: str) -> str:
    """Clean up model name for nicer display in LaTeX."""
    # Example: 'Att_16_4_2_nyc_bike_rownorm' -> 'Att 16 4 2 rownorm'
    name = name.replace("_nyc", "").replace("_chi", "")
    name = name.replace("collab", "").replace("only", "")
    name = name.replace("FedME", "SplitST")
    parts = name.split("_")
    return " ".join([p for p in parts if p not in ["road", "bike", "ped", "taxi", "minmax"]])

def highlight_best_second(values, is_lower_better=True):
    """Return dict of formatted LaTeX strings with bold/underline for best/second best."""
    arr = np.array(values, dtype=float)
    order = np.argsort(arr) if is_lower_better else np.argsort(-arr)
    
    formatted = [f"{v:.2f}" for v in arr]
    if len(arr) > 0:
        formatted[order[0]] = f"\\textbf{{{formatted[order[0]]}}}"
    if len(arr) > 1:
        formatted[order[1]] = f"\\underline{{{formatted[order[1]]}}}"
    return formatted

def make_dynamic_header(df):
    """
    Build a LaTeX header with two rows:
    - first row: mode names (everything before _MAE/_RMSE)
    - second row: metric names (MAE, RMSE, ...)
    """
    # Extract base modes and metrics
    modes = []
    metrics = []
    for col in df.columns:
        if "_" in col:
            base, metric = col.rsplit("_", 1)
        else:
            base, metric = col, ""
        modes.append(base.replace("_", " "))
        metrics.append(metric)

    # First row: Model Name + multicolumns for each mode
    first_row = "Model Name"
    col_format = "l"
    i = 0
    while i < len(modes):
        base = modes[i]
        # count how many consecutive columns belong to same base
        count = 1
        while i + count < len(modes) and modes[i + count] == base:
            count += 1
        col_format += "|" + "c" * count
        first_row += f" & \\multicolumn{{{count}}}{{c|}}{{{base}}}"
        i += count
    first_row += " \\\\"

    # Second row: metric names under each mode
    second_row = " "
    for metric in metrics:
        second_row += f" & {metric}"
    second_row += " \\\\"

    return first_row, second_row, col_format


def plot_table_all_modes(models, results_file="results/results.json"):
    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Collect results into structured dict
    table = {}
    for mode, model_list in models.items():
        for model_name in model_list:
            # Skip if model name contains "only"
            if "only" in model_name:
                continue
            if model_name in all_results:
                metrics = all_results[model_name]["metrics"]
                #mode_key = f"{mode}_MAE" if "MAE" in metrics else f"{mode}_RMSE"
                
                # generalize the model name for table
                clean_name = format_model_name(model_name)
                
                if clean_name not in table:
                    table[clean_name] = {}
                table[clean_name][f"{mode}_MAE"] = metrics.get("MAE", "N/A")
                table[clean_name][f"{mode}_RMSE"] = metrics.get("RMSE", "N/A")

    df = pd.DataFrame.from_dict(table, orient="index")
    df = df.round(2)
    # replace _ with space in first row
    
    # df replace "rowwise" by $^{*}$

    df = df.rename(index=lambda x: x.replace("stgcn", "STGCN").replace("LSTM DSTGCRN", "LSTM-CSV").replace("FGCRN", "F-GCRN"))
    # if SplitST and True in cell -> remove True and replace SplitST by FedSplitST
    # if SplitST and False in cell -> remove False and keep SplitST
    def clean_cell(x):
        if not isinstance(x, str):
            return x
        #print(x)
        if isinstance(x, str):
            if "SplitST" in x and "True" in x:
                return x.replace("SplitST", "FedSplitST").replace("True", "").strip()
            elif "SplitST" in x and "False" in x:
                return x.replace("False", "").strip()
        return x

    #df = df.map(clean_cell)
    df.index = df.index.map(clean_cell)  
    df = df.rename(index=lambda x: x.replace("rowwise", "$^{*}$").replace("rownorm", "$^{*}$").replace("norm",""))  
        
    # Sort rows att *, stgcn *, SplitST, SplitST * att, F-GCRN*, pFedCTP, LSTM, FedSplit att, FedSplit * att
    # the others can be removed
    def sort_key(name):
        if "Att" in name and "*" in name:
            return (0, name)
        elif "STGCN" in name and "*" in name:
            return (1, name)
        elif "SplitST" in name and "att" in name and "*" in name and "Fed" not in name:
            return (2, name)
        elif "SplitST" in name and "att" in name and "Fed" not in name:
            return (3, name)
        elif "F-GCRN" in name and "*" in name:
            return (4, name)
        elif "pFedCTP" in name:
            return (5, name)
        elif "LSTM" in name:
            return (6, name)
        elif "FedSplitST" in name and "att" in name:
            return (7, name)
        elif "FedSplitST" in name and "att" in name and "*" in name:
            return (8, name)
        else:
            # drop that row
            return (9, name)  # Default case
    df = df.reindex(sorted(df.index, key=sort_key))
    # drop all rows that have 9 as first element in sort_key
    df = df[df.index.map(lambda x: sort_key(x)[0] != 9)]
            
    

    # Highlight best and second-best for each column
    for col in df.columns:
        formatted = highlight_best_second(df[col].tolist())
        df[col] = formatted

    # Convert to LaTeX IEEE format
    latex = df.to_latex(
        escape=False,
        column_format="l|cc|cc|cc",
        multicolumn=True,
        multicolumn_format="c",
        header=True
    )
    
    first_row, second_row, col_format = make_dynamic_header(df)

    latex = df.to_latex(
        escape=False,
        column_format=col_format,
        header=False  # we’ll inject our own header
    )

    # Inject dynamic header
    latex = latex.replace(
        "\\toprule",
        "\\toprule\n" + first_row + "\n" + second_row + "\n\\midrule"
    )

    print(latex)
    return latex


def plot_ablation_study(models, results_file="results/results.json"):
    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Collect results into structured dict
    table = {}
    for mode, model_list in models.items():
        for model_name in model_list:
            if "only" not in model_name:
                    continue
            #print(model_name)
            if model_name in all_results:
                metrics = all_results[model_name]["metrics"]
                #clean_name = model_name
                clean_name = format_model_name(model_name)
                #print(f" {model_name} -> {clean_name}")
                if clean_name not in table:
                    table[clean_name] = {}
                table[clean_name][f"{mode}_MAE"] = metrics.get("MAE", "N/A")
                table[clean_name][f"{mode}_RMSE"] = metrics.get("RMSE", "N/A")
            """ 
                        if model_name in all_results:
                metrics = all_results[model_name]["metrics"]
                #mode_key = f"{mode}_MAE" if "MAE" in metrics else f"{mode}_RMSE"
                
                # generalize the model name for table
                clean_name = format_model_name(model_name)
                
                if clean_name not in table:
                    table[clean_name] = {}
                table[clean_name][f"{mode}_MAE"] = metrics.get("MAE", "N/A")
                table[clean_name][f"{mode}_RMSE"] = metrics.get("RMSE", "N/A")"""

    df = pd.DataFrame.from_dict(table, orient="index")
    df = df.round(2)
    
    # sort rows such that first ((no "rowwise" in name) then "rowwise") -> spatial, temporal False, temporal True
    def sort_key(name):
        if "rowwise" not in name:
            if "spatial" in name:
                return (0, name)
            elif "temporal" and "False" in name:
                return (1, name)
            elif "temporal" and "True" in name:
                return (2, name)
        elif "rowwise" in name:
            if "spatial" in name:
                return (3, name)
            elif "temporal" and "False" in name:
                return (4, name)
            elif "temporal" and "True" in name:
                return (5, name)
        return (6, name)  # Default case
    
    df = df.reindex(sorted(df.index, key=sort_key))
    
    def clean_cell(x):
        if not isinstance(x, str):
            return x
        #print(x)
        if isinstance(x, str):
            if "SplitST" in x and "True" in x:
                return x.replace("SplitST", "FedSplitST").replace("True", "").strip()
            elif "SplitST" in x and "False" in x:
                return x.replace("False", "").strip()
        return x

    #df = df.map(clean_cell)
    df.index = df.index.map(clean_cell)  
    df = df.rename(index=lambda x: x.replace("rowwise", "$^{*}$").replace("rownorm", "$^{*}$").replace("norm",""))  

    # Highlight best and second-best for each column
    for col in df.columns:
        formatted = highlight_best_second(df[col].tolist())
        df[col] = formatted
        
        # Convert to LaTeX IEEE format
    latex = df.to_latex(
        escape=False,
        column_format="l|cc|cc|cc",
        multicolumn=True,
        multicolumn_format="c",
        header=True
    )
    
    first_row, second_row, col_format = make_dynamic_header(df)

    latex = df.to_latex(
        escape=False,
        column_format=col_format,
        header=False  # we’ll inject our own header
    )

    # Inject dynamic header
    latex = latex.replace(
        "\\toprule",
        "\\toprule\n" + first_row + "\n" + second_row + "\n\\midrule"
    )

    print(latex)
    return latex
    
    
def print_table(mode, mode_model, results_file):
    now = datetime.now()
    now = now.strftime("%y%m%d_%H%M%S")
    """
    Print a table to show the results of all models
    One row per model with each metric in a column
    
    """
    # import json file
    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Only keep the json entries of the models in mode_model
    all_results = {k: v for k, v in all_results.items() if k in mode_model}
    
    # Iterate through all results and take the values of the key "metrics" and extract the values to a table
    table = []
    for model_name, model_results in all_results.items():
        metrics = model_results["metrics"]
        # Add the model name to the metrics
        metrics["model_name"] = model_name
        
        # Check if the key number_of_parameters exists
        if "number_of_parameters" in model_results:
            metrics["number_of_parameters"] = model_results["number_of_parameters"]
        else:
            metrics["number_of_parameters"] = "N/A"

        #print(metrics)
        # Add the metrics to the table
        table.append(metrics)
    
    # Create a pandas dataframe from the table
    df = pd.DataFrame(table)
    # reduce to 2 decimal places
    df = df.round(2)
    
    # rename model name and set as first column
    df.rename(columns={"model_name": "Model Name"}, inplace=True)
    df.set_index("Model Name", inplace=True)
    
    df.index = df.index.str.replace("_", " ")  # Replace underscores with spaces in index

    
    # Store as latex table, with title of the traffic mode, horizontal lines and vertical lines
    # Store as latex table
    #df.to_latex(f"results/tables/{now}_{mode}.tex", index=True, float_format="%.2f", escape=False)
    df.to_latex(
        f"results/tables/{now}_{mode}.tex",
        index=True,
        float_format="%.2f",
        escape=False,
        caption=f"Results for {mode} mode",
        label=f"tab:{mode}",
        column_format="|l|" + "c|" * len(df.columns),
        #hrules=True  # Only works in pandas >= 2.0
    )
    
    print(df)
    
    


def get_model_names(path):
    """
    Get the model names and their corresponding JSON files.
    :return: Dictionary of model names and their JSON files.
    """
    
    #path = "results_server/results_2025-05-19_18-26-11.json"
    
    # Iterate through the JSON file and get all model names and store these as list (model_name: ....)
    
    # Open the JSON file and get all model names
    if not os.path.exists(path):
        print(f"Results file {path} not found.")
        return
    # Load the JSON file
    with open(path, "r") as f:
        data = json.load(f)
        model_names = list(data.keys())
        
    # drop if it doesn't contain either "road", "bike" or "pedestrian"
    model_names = [name for name in model_names if "road" in name or "bike" in name or "ped" in name or "taxi" in name]
    # Drop if it doesn't contain "norm" or "raw"
    #model_names = [name for name in model_names if "norm" in name or "raw" in name or "pFedCTP" in name]
    
    # Drop if "test" in name
    model_names = [name for name in model_names if "test" not in name]
        
    print(f"Model names: {model_names}")
    
    return model_names
    
def plot_single_predictions(n_samples, model_labels, mode, path):
    # Get y_true values
    y_true = np.load(f"{path}/{mode}_test_Y.npy")
    print(f"y_true shape: {y_true.shape}")
    
    # Get the values
    data_path = "results_server/predictions/"
    y_pred = []
    # in folder results_server/predictions get the npz file with the model name
    for model in model_labels:
        try:
            file_path = f"{data_path}/{model}_{mode}_predictions.npz"
        except:
            file_path = f"{data_path}/{model}_predictions.npz"
        print(f"Loading from {file_path}")
        if os.path.exists(file_path):
            loaded = np.load(file_path, allow_pickle=True)
            arr = loaded["Y_pred"]
            # Check if loaded["y_true"] is equal to y_true
            if not np.array_equal(loaded["Y_true"], y_true):
                print(f"Warning: y_true in {file_path} does not match the global y_true.")
                print(f"Loaded y_true shape: {loaded['Y_true'].shape}, Global y_true shape: {y_true.shape}")
                continue
            
            #print(f"Model {model} predictions shape: {arr.shape}")
            y_pred.append(arr)
        else:
            # Drop model_name from model_labels
            model_labels.remove(model)
            print(f"Model {model} not found in {data_path}.")
            continue
        
    # A radom generator of n tuples (x, y) with x in [0, n_time_windows) and y in [0, n_sensors)
    n_time_windows = y_true.shape[0]
    n_sensors = y_true.shape[2]
    
    np.random.seed(42)
    samples_to_plot = []
    for i in range(n_samples):
        x = np.random.randint(0, n_time_windows)
        y = np.random.randint(0, n_sensors)
        samples_to_plot.append((x, y))
    
    d_time_steps = y_pred[0].shape[1]
    time_steps = np.arange(d_time_steps)
        
        
    # Create the subplots (4 rows, 1 column)
    rows = math.ceil(n_samples / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4*rows), sharex=True)
    axes = axes.ravel()  # flatten the 2D axes array
    
    # Loop through the samples and plot each one
    for i, tuple_value in enumerate(samples_to_plot):
        ax = axes[i]
        sample_idx, sensor_idx = tuple_value  # Unpack the tuple
        #print(f"Sample index: {sample_idx}, Sensor index: {sensor_idx}")
        true_values = y_true[sample_idx, :, sensor_idx]  # True values for the i-th selected sample
        # Plot true values in black
        ax.plot(time_steps, true_values, label='True Values', color='black', lw=2)
        
        # Plot predicted values from all models in different colors
        for j, pred in enumerate(y_pred):
            pred_values = pred[sample_idx, :, sensor_idx]
            if model_labels:
                ax.plot(time_steps, pred_values, lw=1, label=model_labels[j])
            else:
                ax.plot(time_steps, pred_values, lw=1)

        # Set labels and title for each subplot
        ax.set_xlabel('Time Steps (Hours)')
        ax.set_ylabel('Flow')
        ax.set_title(f'Sensor idx: {sensor_idx} time window: {sample_idx}', fontsize=10)  # sensor idx

    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    # Add a single legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()  # Get the handles and labels of the last plot
    # place legend at bottom center
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=4,
        fontsize=10,
        frameon=False, 
    )
    #plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    # Overall title
    plt.suptitle(f"Predictions for {mode} mode - {n_samples} samples", fontsize=12)

    # now as yymmdd-hhmmss
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    
    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(f"results/plots/predictions_{mode}_{now}.png", dpi=300)
    plt.show()
    
def print_single_prediction(model):
    # Open the npz file with the predictions
    data_path = "results_server/predictions/"
    file_path = data_path + model + "_predictions.npz"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    loaded = np.load(file_path)
    y_pred = loaded["Y_pred"]
    
    n_time_windows = y_pred.shape[0]
    n_sensors = y_pred.shape[2]
    
    sample_idx = np.random.randint(0, n_time_windows)
    sensor_idx = np.random.randint(0, n_sensors)
    
    #sample_idx, sensor_idx = tuple_value
    pred_values = y_pred[sample_idx, :, sensor_idx]
    
    
    print(f"Sample index: {sample_idx}, Sensor index: {sensor_idx}")
    print(f"Predicted values: {pred_values}")
    
    print(y_pred[sample_idx+1, :, sensor_idx])
    

def compare_models_results(model_labels, mode, path):
    # Get y_true values
    y_true = np.load(f"{path}/{mode}_test_Y.npy")
    
    # Get the values
    data_path = "results_server/predictions/"
    y_pred = []
    # in folder results_server/predictions get the npz file with the model name
    for model in model_labels:
        file_path = f"{data_path}/{model}_{mode}_predictions.npz"
        if os.path.exists(file_path):
            loaded = np.load(file_path, allow_pickle=True)
            arr = loaded["Y_pred"]
            # Check if loaded["y_true"] is equal to y_true
            if not np.array_equal(loaded["Y_true"], y_true):
                print(f"Warning: y_true in {file_path} does not match the global y_true.")
                print(f"Loaded y_true shape: {loaded['Y_true'].shape}, Global y_true shape: {y_true.shape}")
                continue
            
            
            #print(f"Model {model} predictions shape: {arr.shape}")
            y_pred.append(arr)
        else:
            # Drop model_name from model_labels
            model_labels.remove(model)
            print(f"Model {model} not found in {data_path}.")
            continue
    
    y_pred = np.array(y_pred)  # shape (n_models, n_samples, horizon, n_sensors)
    print(y_pred.shape)
    print(y_true.shape)
    
    # for each model state the number of over and underestimations
    # Iterate over first dimension of y_pred
    for i, model in enumerate(model_labels):
        preds = y_pred[i]  # shape (n_samples, horizon, n_sensors)
        overestimations = np.sum(preds > y_true)
        underestimations = np.sum(preds < y_true)
        exact_predictions = np.sum(preds == y_true)
        total_predictions = preds.size
        print(f"Model: {model} - {overestimations/total_predictions*100:.2f}% overestimations, {underestimations/total_predictions*100:.2f}% underestimations, {exact_predictions/total_predictions*100:.2f}% exact predictions")
    
    # Check the mae and rmse per model and for each prediction time step
    # initialize a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        errors = np.abs(preds - y_true)
        # avg over the second dimension (horizon)
        mean_errors = np.mean(errors, axis=(0, 2))
        # add plot to existing figure
        ax.plot(mean_errors, label=model)
    ax.set_xlabel("Prediction Time Step")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Mean Absolute Error over Prediction Horizon for {mode} mode")
    ax.legend()
    plt.savefig(f"results_server/results_analysis/horizon_mae_{mode}.png", dpi=300)
    plt.close()
        
    # Check the error distribution per model and sensor
    # I want a plot with sensors on the x axis and mean error per sensor on the y axis -> one point for the mean error per sensor and model
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        errors = np.abs(preds - y_true)     
        # avg over the first two dimensions (samples and horizon)
        mean_errors = np.mean(errors, axis=(0, 1))        
        # add plot to existing figure
        ax.plot(mean_errors, label=model)
        
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Mean Absolute Error per Sensor for {mode} mode")
    ax.legend()
    plt.savefig(f"results_server/results_analysis/sensor_mae_{mode}.png", dpi=300)
    # delete plt
    plt.close()
    
    # same with relative error
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        # mask 0
        mask_0 = y_true == 0
        errors = np.abs(preds - y_true) / (np.abs(y_true) + 1e-6)
        # replace mask_0 from errors by 0
        #errors[mask_0] = 0
        # avg over the first two dimensions (samples and horizon)
        mean_errors = np.mean(errors, axis=(0, 1))
        # add plot to existing figure
        ax.plot(mean_errors, label=model)
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Mean Relative Absolute Error")
    ax.set_title(f"Mean Relative Absolute Error per Sensor for {mode} mode")
    ax.legend()
    plt.savefig(f"results_server/results_analysis/sensor_relative_mae_{mode}.png", dpi=300)
    plt.close()
    
    # print the index of the 10 sensors with the highest mean error
    print(f"Model: {model} - Top 10 sensors with highest mean error: \t{np.argsort(-mean_errors)[:10]}")
    
    # Check the error development over the batch samples
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        errors = np.abs(preds - y_true)     
        # avg over the last two dimensions (horizon and sensors)
        mean_errors = np.mean(errors, axis=(1, 2))        
        # add plot to existing figure
        ax.plot(mean_errors, label=model)
    ax.set_xlabel("Batch Sample Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Mean Absolute Error over Batch Samples for {mode} mode")
    ax.legend()
    plt.savefig(f"results_server/results_analysis/batch_sample_mae_{mode}.png", dpi=300)
    plt.close()
    
    
    # Heatmap per model with x_axis time horizon and y_axis sensors
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        errors = np.abs(preds - y_true)     
        # avg over the first dimension (samples)
        mean_errors = np.mean(errors, axis=0)
        plt.figure(figsize=(12, 6))
        plt.imshow(mean_errors.T, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Absolute Error')
        plt.xlabel("Prediction Time Step")
        plt.ylabel("Sensor Index")
        plt.title(f"Mean Absolute Error Heatmap for {model} in {mode} mode")
        plt.savefig(f"results_server/results_analysis/heatmap_mae_{model}_{mode}.png", dpi=300)
    
    
    
    # For each model and mode get the error distribution (absolute) and plot it as curve

def locate_errors(model_labels, mode, path):
    # Get y_true values
    y_true = np.load(f"{path}/{mode}_test_Y.npy")
    
    # Get the values
    data_path = "results_server/predictions/"
    y_pred = []
    # in folder results_server/predictions get the npz file with the model name
    for model in model_labels:
        file_path = f"{data_path}/{model}_{mode}_predictions.npz"
        if os.path.exists(file_path):
            loaded = np.load(file_path, allow_pickle=True)
            arr = loaded["Y_pred"]
            # Check if loaded["y_true"] is equal to y_true
            if not np.array_equal(loaded["Y_true"], y_true):
                print(f"Warning: y_true in {file_path} does not match the global y_true.")
                print(f"Loaded y_true shape: {loaded['Y_true'].shape}, Global y_true shape: {y_true.shape}")
                continue
            
            
            #print(f"Model {model} predictions shape: {arr.shape}")
            y_pred.append(arr)
        else:
            # Drop model_name from model_labels
            model_labels.remove(model)
            print(f"Model {model} not found in {data_path}.")
            continue
        
    # Compute the error
    y_pred = np.array(y_pred)  # shape (n_models, n_samples, horizon, n_sensors)
    err = np.abs(y_pred - y_true)  # shape (n_models, n_samples, horizon, n_sensors)
    
    
    # Get the avg error over all models
    err_mean = np.mean(err, axis=0)  # shape (n_samples, horizon, n_sensors)
    err_mean = err[3]  # just take the first model for now
    # Get the index of the top 2 sensors with the highest error
    top_2_sensors_idx = np.argsort(-np.mean(err_mean, axis=(0, 1)))[:2]
    
    # For both sensors plot heatmap with batches in y ans x axis time horizon
    for sensor_idx in top_2_sensors_idx:
        plt.figure(figsize=(12, 6))
        plt.imshow(err_mean[:, :, sensor_idx], aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Absolute Error')
        plt.xlabel("Prediction Time Step")
        plt.ylabel("Batch Sample Index")
        plt.title(f"Absolute Error Heatmap {model_labels[3]} for Sensor {sensor_idx} in {mode} mode")
        plt.savefig(f"results_server/results_analysis/top_sensor_{sensor_idx}_heatmap_{mode}.png", dpi=300)
        
        # For these sensors plot the y_true
        plt.figure(figsize=(12, 6))
        plt.imshow(y_true[:, :, sensor_idx], aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='True Value')
        plt.xlabel("Prediction Time Step")
        plt.ylabel("Batch Sample Index")
        plt.title(f"True Values for Sensor {sensor_idx} in {mode} mode")
        plt.savefig(f"results_server/results_analysis/top_sensor_{sensor_idx}_true_values_{mode}.png", dpi=300)
    
    
    
    
    # Get a plot of hte cumulative error distribution for each model
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_idx, model in enumerate(model_labels):
        errors = err[model_idx].flatten()
        sns.kdeplot(errors, cumulative=True, label=model, ax=ax)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution for {mode} mode (total samples: {errors.shape[0]})")
    ax.legend()
    plt.savefig(f"results_server/results_analysis/error_distribution_{mode}.png", dpi=300)
    
    # print the error values at 50%, 75%, 90%, 95%, 99%
    for model_idx, model in enumerate(model_labels):
        errors = err[model_idx].flatten()
        percentiles = [75, 90, 95, 99]
        values = np.percentile(errors, percentiles)
        #print(f"Model: {model} - Error percentiles:")
        for p, v in zip(percentiles, values):
            #print(f"  {p}th percentile: {v:.2f}")
            continue
        # K = value at 1% percentile
        K = int(np.percentile(errors, 1))
    
    # L largest errors at the 5% percentile
    L = 2 * (y_true.shape[0] * y_true.shape[1] * y_true.shape[2]) // 100
    flat_err = err.flatten()
    flat_err = torch.tensor(flat_err)
    values, indices = torch.topk(flat_err, L)

    # convert flat indices back to (B,H,N)
    B, H, N = err[0].shape
    b_idx = indices // (H * N)
    hn = indices % (H * N)
    h_idx = hn // N
    n_idx = hn % N
    
    # For each model find the samples where the absolute error is greater than a threshold (e.g., 100)
    threshold = K
    for model_idx, model in enumerate(model_labels):
        preds = y_pred[model_idx]
        errors = np.abs(preds - y_true)
        indices = np.where(errors > threshold)
        print(f"Model: {model} - Number of errors above {threshold}: {len(indices[0])}")
        # Print first 10 indices
        for i in range(min(10, len(indices[0]))):
            print(f" Sample: {indices[0][i]}, Time step: {indices[1][i]}, Sensor: {indices[2][i]}, Error: {errors[indices[0][i], indices[1][i], indices[2][i]]}")
            

def get_context_of_errors(model_labels, mode, path, weather, kalendar):
    
    df_weather = weather
    # only keep int(0.85 * df_weather.shape[0])+1: last rows
    df_weather = df_weather.iloc[int(0.85 * df_weather.shape[0])+1: , :]
    
    # over all column only keep the prefix main_ and group all columns (mean) with same prefix
    num = df_weather.select_dtypes(include="number").groupby(lambda x: x.split("_")[0], axis=1).mean()
    non_num = df_weather.select_dtypes(exclude="number")
    df_weather_grouped = non_num.join(num)
    print(f"From weather original shape: {df_weather.shape} to grouped shape: {df_weather_grouped.shape}")
    
    weather = df_weather.to_numpy()
    print(f"Weather shape after filtering: {weather.shape}")
    
    df_kalendar = kalendar
    df_kalendar = df_kalendar.iloc[int(0.85 * df_kalendar.shape[0])+1: , :]
    # simplify kalendar names
    columns_changes = {
        "Zurich Film Festival (ZFF)" : "ZFF",
        'UCI Road World Championships (region Zurich)' : 'UCI WC',
        'Major Stadium Concerts' : "Lezi Concert",
        'National Day (Bundesfeier)' : "1. Aug",
        'Summer Holidays' : "Summer H",
        'Street Parade' : "SP",
    }
    df_kalendar = df_kalendar.rename(columns=columns_changes)
    # delete column any_holidays
    df_kalendar = df_kalendar.drop(columns=["any_holiday"])
    # replace all Holiday by H in all columns

    # drop columns weekday, hour, month, is_holiday
    #df_kalendar = df_kalendar.drop(columns=["weekday", "hour", "month", "is_holiday"])
    kalendar = df_kalendar.to_numpy()
    #kalendar = kalendar[int(0.85 * df_kalendar.shape[0])+1:, :]
    
    # get y_true values
    y_true = np.load(f"{path}/{mode}_test_Y.npy")
    y_pred = []
    # get X values
    
    # init pdf
    pdf_tables = PdfPages(f"results/plots/error_context_{mode}.pdf")
    pdf_plots = PdfPages(f"results/plots/error_context_plots_{mode}.pdf")
    
    valid_models = []
    for model in model_labels:
        data_path = "results_server/predictions/"
        
        if mode in model:
            file_path = f"{data_path}/{model}_predictions.npz"
        else:
            file_path = f"{data_path}/{model}_{mode}_predictions.npz"
        if os.path.exists(file_path):
            loaded = np.load(file_path, allow_pickle=True)
            arr = loaded["Y_pred"]
            # Check if loaded["y_true"] is equal to y_true
            if not np.array_equal(loaded["Y_true"], y_true):
                print(f"Warning: y_true in {file_path} does not match the global y_true.")
                print(f"Loaded y_true shape: {loaded['Y_true'].shape}, Global y_true shape: {y_true.shape}")
                continue
            #print(f"Model {model} predictions shape: {arr.shape}")
            y_pred.append(arr)
            valid_models.append(model)
            
        else:
            print(f"Model {model} not found in {data_path}.")
            continue
    model_labels = valid_models    
    
    y_pred = np.array(y_pred)  # shape (n_models, n_samples, horizon, n_sensors)
    err = np.abs(y_pred - y_true)  # shape (n_models, n_samples, horizon, n_sensors)
    y_pred_copy = y_pred
    
    horizon = y_pred.shape[2]
    print(f"Horizon: {horizon}")
    
    # reshape err to (n_models, n_samples * horizon, n_sensors)
    err = err.reshape(err.shape[0], -1, err.shape[3])
    y_true = y_true.reshape(-1, y_true.shape[2])
    y_pred = y_pred.reshape(y_pred.shape[0], -1, y_pred.shape[3])
    
    # Check if the shape of weather and agenda match the second and third dimension of err_reshaped
    if weather.shape[0] != err.shape[1]:
        print(f"Weather shape {weather.shape} does not match error shape {err.shape}.")
        return
    top = 50
    # Get the biggest errors for each model
    # use the index and check in the weather and agenda data what was the context
    for model_idx, model in enumerate(model_labels):
        errors = err[model_idx]
        pred = y_pred[model_idx]
        # Get the indices (tuples) of the top 10 errors
        idx = np.argpartition(errors.ravel(), -top)[-top:]
        rows, cols = np.unravel_index(idx, errors.shape)
        top_indices = list(zip(rows, cols))
        
        # Get the corresponding error values
        top_values = errors[rows, cols]
        
        # Sort descending by error
        sorted_idx = np.argsort(-top_values)
        top_indices = [top_indices[i] for i in sorted_idx]
        
        get_context=False
        get_table_context=True
        if get_table_context == True:
            init_table = pd.DataFrame()
            
        print(f"\n--- Model: {model} for {mode} ---")
        #####################################
        ## Here I want to iterate over the top indices and print the context (index tuples of time_stamp and sensor)
        ## print( the timestamp in date and hour and also the main weather condition and kalendar context)
        ## The weather and kalendar states need to be filtered to show the relevant info only
        for idx in top_indices:
            
            # Check index
            val_y = y_pred[model_idx, idx[0], idx[1]]
            val_y_pd = y_pred_copy[model_idx, idx[0]//horizon, idx[0]%horizon, idx[1]]
            if not np.isclose(val_y, val_y_pd):
                print(f"Warning: Value mismatch between reshaped and original y_pred - {val_y} vs {val_y_pd}")
            
            weekday = df_kalendar.iloc[idx[0]]["weekday"]
            weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(weekday)]
            #print(f"Date: {kalendar[idx[0], 0]} ({weekday}) Sensor: {idx[1]} - Y_true: {y_true[idx[0], idx[1]]} - Y_pred: {pred[idx[0], idx[1]]:.2f} with Error: {errors[idx[0], idx[1]]:.2f}")
            #print(f"Error: {errors[idx[0], idx[1]]:.2f} at Sample-TimeStep: {idx[0]}, Sensor: {idx[1]}")
            kalendar_context = kalendar[idx[0],:]
            #print(agenda_context)
            if get_context == True:
                # get the indices where kalendar_context is 1
                vals = pd.to_numeric(kalendar_context, errors="coerce")
                idx_with_1 = np.where(vals > 0)[0]
                # remove if last 4 indices (weekday, hour, month, is_holiday)
                idx_with_1 = [i for i in idx_with_1 if i < len(df_kalendar.columns)-4]
                
                # get the attribute from kalendar.columns
                kalendar_features = [df_kalendar.columns[i] for i in idx_with_1 if i < len(df_kalendar.columns)]
                print(f" > Events: {kalendar_features}, with values: {kalendar_context[idx_with_1]}")
                
                # get the main weather condition
                weather_context = weather[idx[0], :]
                
                # fkl010h0 wind speed hourly mean m/s
                # fkl010h3 Gust peak (three seconds); hourly maximum in m/s
                # htoauths Snow depth (automatic measurement); hourly current value
                # rre150h0 Precipitation; hourly total
                # sre000h0 Sunshine duration; hourly total
                # ure200h0 Relative humidity; hourly mean in %
                # tre200h0 Air temperature 2 m above ground; hourly mean
                # tre005h0 Air temperature at 5 cm above grass; hourly mean
                # get the columns index based on the df_weather of these features
                weather_features_of_interest = {
                    "fkl010h0": "Wind Speed (m/s)",
                    "fkl010h3": "Gust Peak (m/s)",
                    "htoauths": "Snow Depth (cm)",
                    "rre150h0": "Precipitation (mm)",
                    "sre000h0": "Sunshine Duration (hours)",
                    "ure200h0": "Relative Humidity (%)",
                    "tre200h0": "Air Temperature 2m (°C)",
                    "tre005h0": "Air Temperature 5cm (°C)"
                }
                weather_features = df_weather.columns
                # get a list with the index of the columns in the keys of weather_features_of_interest
                idx_weather_features = [i for i, col in enumerate(weather_features) if col.startswith(tuple(weather_features_of_interest.keys()))]
                # get a dict with key of weather feature of interest name and all values from weather_context that start with the keys of weather_features_of_interest
                idx_weather_features_dict = {weather_features_of_interest[weather_features[i].split("_")[0]]: weather_context[i] for i in idx_weather_features}
                print(f" > Weather: {idx_weather_features_dict}")
                
            if get_table_context == True:
                # get a table with entry per row, columns for Date, Sensor, Y_true, Y_pred, Error, Kalendar Events (list of events if not empty), Weather Features (temperature, peak gust, precipitation, snow))
                date = kalendar[idx[0], 0]
                time_index = idx[0] % horizon
                weekday = df_kalendar.iloc[idx[0]]["weekday"]
                sensor = idx[1]
                y_t = y_true[idx[0], idx[1]]
                y_p = pred[idx[0], idx[1]]
                error = errors[idx[0], idx[1]]
                # kalendar events
                kalendar_context = kalendar[idx[0],:]
                vals = pd.to_numeric(kalendar_context, errors="coerce") if kalendar is not None else []
                
                # weather features
                # in df_weather get the mean of all cells in row idx[0] for column that contain fkl010h0
                wind_speed = df_weather_grouped.iloc[idx[0]]["fkl010h0"]
                snow_depth = df_weather_grouped.iloc[idx[0]]["htoauths"]
                precipitation = df_weather_grouped.iloc[idx[0]]["rre150h0"]
                air_temp_2m = df_weather_grouped.iloc[idx[0]]["tre200h0"]
                
                # make table row
                table_row = {
                    "Date": date.split(" ")[0],
                    "Hour": date.split(" ")[1],
                    "Time index": time_index+1,
                    "Weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(weekday)],
                    "Sensor": sensor,
                    "Y_true": y_t,
                    "Y_pred": f"{y_p:.2f}",
                    "Error": f"{error:.2f}",
                    "Kalendar Events": ', '.join([df_kalendar.columns[i] for i in np.where(vals > 0)[0] if i < len(df_kalendar.columns)-4]) if kalendar is not None else "N/A",
                    "Wind (m/s)": f"{wind_speed:.2f}" if weather is not None else "N/A",
                    "Snow (cm)": f"{snow_depth:.2f}" if weather is not None else "N/A",
                    "Precipitation (mm)": f"{precipitation:.2f}" if weather is not None else "N/A",
                    "Air Temp": f"{air_temp_2m:.2f}" if weather is not None else "N/A"
                }
                
                init_table = pd.concat([init_table, pd.DataFrame([table_row])], ignore_index=True)
        
        if top < 41:    
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            ax.set_title(f"Error Contexts for {model} in {mode} mode", fontsize=14, pad=20)
            
            tbl = ax.table(
                cellText=init_table.values,
                colLabels=init_table.columns,
                cellLoc='center',
                loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6)
            
            for (row, col), cell in tbl.get_celld().items():
                cell.visible_edges = "horizontal"     # only horizontal lines
                cell.set_edgecolor("0.8")              # light grey
                cell.set_linewidth(0.5)

                if row == 0:                           # header row
                    cell.set_text_props(weight="bold")
            
            pdf_tables.savefig(fig, bbox_inches='tight')
            plt.close()
        
        
        
        #######################################################################
        # Do some recap stats for the entire df per categorie
        # for each of the column get values count and the weighted value count (weighted by the "error" column)
        # columns to consider
        cat_cols = ["Date", "Hour", "Time index", "Weekday", "Sensor", "Kalendar Events"]
        num_cols = ["Y_true", "Wind (m/s)", "Snow (cm)", "Precipitation (mm)", "Air Temp"]
        
        # create a little plot for each column
        n_plots = len(cat_cols) + len(num_cols)
        n_cols = 3
        nr_rows = int(math.ceil(n_plots / n_cols))
        
        fig, axes = plt.subplots(nr_rows, n_cols, figsize=(8.27, 11.69))
        # set title
        fig.suptitle(f"Error Contexts Summary for {model} in {mode} mode (top {top} values)", fontsize=14, y=0.98)
        axes = axes.ravel()
        
        cumsum = False
        
        init_table["Error"] = pd.to_numeric(init_table["Error"], errors='coerce')
        
        i=0
        for col in cat_cols:
            init_table[col] = init_table[col].astype(str)
            
            if cumsum:
                values_counts = init_table.groupby(col)["Error"].sum().sort_values(ascending=False)
                print(values_counts.head())
                if values_counts.shape[0] > 25:
                    values_counts = values_counts.head(25)
                values_counts.plot.bar(ax=axes[i], color='skyblue')
            else:
                value_counts = init_table[col].value_counts()
                if value_counts.shape[0] > 25:
                    value_counts = value_counts.head(25)
                value_counts.plot.bar(ax=axes[i], color='skyblue')
            axes[i].set_title(f"{col}")
            i += 1
            
        for col in num_cols:
            data = pd.to_numeric(init_table[col], errors='coerce').dropna()
            bins = np.histogram_bin_edges(data, bins='auto')
            if cumsum:
                # group by data bin and
                init_table['binned'] = pd.cut(data, bins=bins)
                values_counts = init_table.groupby('binned')["Error"].sum().sort_index()
                if values_counts.shape[0] > 25:
                    values_counts = values_counts.iloc[:25]
                values_counts.plot.bar(ax=axes[i], color='skyblue')
            else:                    
                axes[i].hist(data, bins=bins, color='skyblue', edgecolor='black')
            axes[i].set_title(f"{col}")
            i += 1
        
        # make a side margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05)
        plt.tight_layout()
        
        pdf_plots.savefig(fig, bbox_inches='tight')
        # close plot
        plt.close()
        
    
    pdf_tables.close()
    pdf_plots.close()
            

                
        
                
        

    