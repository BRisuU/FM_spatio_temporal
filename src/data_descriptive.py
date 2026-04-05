import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import squareform, pdist

from scipy.stats import skew, kurtosis, wasserstein_distance
from scipy.signal import periodogram
#from tsfresh.feature_extraction import extract_features
#from catch22 import catch22_all
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
#from antropy import sample_entropy

#from statsmodels.tsa.stattools import acf, pacf
from tqdm import tqdm


def compute_rolling_stats(sensor_series, window=168, step=168):
    stats = {'mean': [], 'var': [], 'skew': [], 'kurt': []}
    for start in range(0, len(sensor_series) - window + 1, step):
        window_data = sensor_series[start:start + window]
        if window_data.isna().mean() > 0.1: continue  # Skip if too sparse
        #stats['mean'].append(window_data.mean())
        #stats['var'].append(window_data.var())
        stats['skew'].append(skew(window_data))
        stats['kurt'].append(kurtosis(window_data))
    return stats



def compute_entropy_acf(sensor_series, max_lag=48):
    values = sensor_series.dropna().values
    acf = [np.corrcoef(values[:-lag], values[lag:])[0,1] if lag < len(values) else np.nan for lag in range(1, max_lag+1)]
    sampen = sample_entropy(values)
    return np.array(acf), sampen

def extract_catch22_features(X_batches):
    # Average features per sensor over all 16h samples
    n_samples, n_sensors, _ = X_batches.shape
    features = np.zeros((n_sensors, 22))
    for sensor in range(n_sensors):
        feats = [catch22_all(X_batches[i, sensor, :])['values'] for i in range(n_samples)]
        features[sensor] = np.nanmean(feats, axis=0)
    return pd.DataFrame(features)

def compute_sensor_similarity(features_df):
    dist_matrix = squareform(pdist(features_df, metric='euclidean'))
    return dist_matrix

def aggregate_dataset_features(list_of_batches):
    dataset_features = []
    for X in list_of_batches:
        feats = extract_catch22_features(X)
        agg = feats.mean(axis=0)
        dataset_features.append(agg)
    return pd.DataFrame(dataset_features)

def compute_wasserstein_matrix(feature_df):
    n = len(feature_df)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = wasserstein_distance(feature_df.iloc[i], feature_df.iloc[j])
            dist_mat[i, j] = dist_mat[j, i] = dist
    return dist_mat

def compute_monthly_features(df):
    # Compute monthly features for entire df
    # Mean, median, std, zeros
    monthly_features = {}
    # Covnert columns to datetime if not already
    df.columns = pd.to_datetime(df.columns, errors='coerce')
    for month in df.columns.to_series().dt.to_period("M").unique():
        month_data = df.loc[:, df.columns.to_series().dt.to_period("M") == month]
        if month_data.empty:
            continue
        monthly_features[month] = {
            'mean': month_data.stack().mean(),
            'median': month_data.stack().median(),
            'std': month_data.stack().std(),
            'zeros': (month_data == 0).sum().sum()
        }
    #print(monthly_features)
    return monthly_features

def get_daily_peak(df):
    # For each sensor in df, compute the daily peak
    # Store the value and the time of the peak (h)
    daily_peaks = {}

    # Convert index to datetime if not already
    df_temp = df.copy()
    df_temp.columns = pd.to_datetime(df_temp.columns, errors='coerce')
    # Resample to daily frequency and get the max value
    
    # Iterate over all rows
    for sensor in df_temp.index:
        sensor_data = df_temp.loc[sensor].dropna()
        if sensor_data.empty:
            continue
        # Resample daily and get both max value and timestamp
        daily_max = sensor_data.resample('W').agg(['max', 'idxmax'])
        daily_peaks[sensor] = {
            'peak_days': daily_max['idxmax'].dt.day_name().values,
            'peak_times': daily_max['idxmax'].values,
            'peak_values': daily_max['max'].values
        }
    
    plt.figure(figsize=(10, 6))
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for sensor, data in daily_peaks.items():
        #peak_days = data['peak_days']      # fixed key names
        #peak_hours = data['peak_hours']
        peak_values = data['peak_values']
        peak_times = pd.to_datetime(data['peak_times'])
        
        # Optional: convert day names to numbers if you want x-axis as weekday
        x_labels = [f"{t.day_name()} {t.hour:02d}h" for t in peak_times]
        
        # Scatter plot: x=hour or x=weekday (depending on choice)
        plt.scatter(x_labels, peak_values, label=sensor)

    plt.xlabel('Hour of Day')
    plt.ylabel('Peak Value')
    plt.title('Weekly Peaks by Hour')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.show()
    
    return daily_peaks

    
    #print(daily_peaks)
        

def compute_descriptive_statistics(df, X_batches, suffix=""):
    
    for name, data in df.items():
        data = data.drop(columns=["FK_STANDORT", "DIRECTION", "ID"], errors='ignore')
        # Daily peak
        get_daily_peak(data)
    
    
    
    df.timeout
    
    # Plot monthly data
    # init monthly features dict
    monthly_features = {}
    for name, data in df.items():
        data = data.drop(columns=["FK_STANDORT", "DIRECTION", "ID"], errors='ignore')
        # Compute descriptive statistics
        monthly_features[name] = compute_monthly_features(data)
    
    # Plot monthly features
    # x axis is month, y axis 1 is mean and std, y axis 2 is zeros
    # one color per df
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()               # Primary axis for mean & std
    ax2 = ax1.twinx()     
    for name, features in monthly_features.items():
        months = list(features.keys())
        
        means = [features[month]['mean'] for month in months]
        stds = [features[month]['std'] for month in months]
        zeros = [features[month]['zeros'] for month in months]
        months = [str(m) for m in features.keys()]
        # Mean + std on primary axis
        ax1.plot(months, means, label=f'{name} Mean')
        ax1.fill_between(months, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.2)

        # Zeros on secondary axis
        ax2.scatter(months, zeros, label=f'{name} Zeros', marker='x')

    # Labels
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Mean / Std')
    ax2.set_ylabel('Zeros')
    
    # rotate x-ticks for better visibility
    ax1.set_xticks(months)
    ax1.set_xticklabels(months, rotation=45)
    
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)  # Adjust y-limits for better visibility
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)  # Adjust y-limits for better visibility

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper center', bbox_to_anchor=(0.5, -0.22),
           ncol=3, frameon=False)

    plt.title('Monthly Features')
    plt.tight_layout()
    plt.savefig(f"data/plots/stats/monthly_features_{suffix}.png", bbox_inches='tight')
        
    
    # Get mean, median, sdt and coefficient of variation for each df
    for name, data in df.items():
        # First column to index and remove it from the data
        # columns to remove if present [FK_STANDORT DIRECTION ID]
        data = data.drop(columns=["FK_STANDORT", "DIRECTION", "ID"], errors='ignore')
            
        # Compute descriptive statistics
        means = data.stack().mean()
        medians = data.stack().median()
        stds = data.stack().std()
        cv = stds / means
        zeros_count = (data == 0).sum().sum()
        total_values = data.size
        zeros_percentage = (zeros_count / total_values) * 100
        #print(f"Descriptive statistics for {name}:")
        #print(f"Means:{means}, Medians:{medians}, Standard deviations:{stds}, Coefficient of variation:{cv}, Zeros count:{zeros_count} {zeros_percentage} \n")
        print(f"{name} & {means.round(2)} & {medians.round(2)} & {stds.round(2)} & {cv.round(2)} & {zeros_count} ({zeros_percentage:.2f}\%) \\\\ \n")
    
    
def compute_temporal_dependence(dfs, lags=[1, 24, 168]):
    """
    dfs: dict of {dataset_name: dataframe of sensors x time}
    lags: list of lags to compute ACF/PACF
    Returns: dict of {dataset: {sensor: acf/pacf values}}
    """
    results = {}

    for name, df in dfs.items():
        print(f"\nComputing temporal dependence for {name} with lags {lags}")
        df = df.drop(columns=["FK_STANDORT", "DIRECTION", "ID"], errors='ignore')
        dataset_res = {}
        for sensor in tqdm(df.index, desc=f'Processing {name}'):
            ts = df.loc[sensor].dropna()
            if len(ts) == 0:
                continue
            acf_vals = acf(ts, nlags=max(lags), fft=False)
            pacf_vals = pacf(ts, nlags=max(lags))
            acf_lags = {f'acf_{l}': acf_vals[l] for l in lags if l < len(acf_vals)}
            pacf_lags = {f'pacf_{l}': pacf_vals[l] for l in lags if l < len(pacf_vals)}
            dataset_res[sensor] = {**acf_lags, **pacf_lags}
        results[name] = dataset_res
        
    plot_temporal_dependence(results, lags=lags)
    return results


def plot_temporal_dependence(results, lags=[1,24,168,720]):
    """
    results: output of compute_temporal_dependence
    Creates subplots for each lag with boxplots colored by dataset
    """
    metrics = ['acf', 'pacf']
    n_lags = len(lags)
    fig, axes = plt.subplots(2, n_lags, figsize=(4*n_lags, 8), sharey='row')

    colors = plt.cm.tab10.colors  # for datasets
    dataset_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        for j, lag in enumerate(lags):
            ax = axes[i, j] if n_lags > 1 else axes[i]
            data_to_plot = []
            for k, dataset in enumerate(dataset_names):
                values = [sensor_res[f'{metric}_{lag}'] 
                          for sensor_res in results[dataset].values() 
                          if f'{metric}_{lag}' in sensor_res]
                data_to_plot.append(values)

            # Boxplot: one per dataset
            bplot = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            # Titles and labels
            ax.set_title(f'{metric.upper()} lag {lag}h', fontsize=15)
            ax.set_xticks(range(1, len(dataset_names)+1))
            # Replace spaces in xlabels with newlines
            ax.set_xticklabels([name.replace(" ", "\n") for name in dataset_names],
                               rotation=45, fontsize=14)

            if j == 0:
                ax.set_ylabel(metric.upper(), fontsize=14)

            # Remove x-axis labels for the top row
            if i == 0:
                ax.set_xticklabels([])

            # Increase tick label fontsize
            ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()
            
        
    """
    
    
    # Rolling stats for one sensor
    #df = df.T
    print(df.head())
    #stats = compute_rolling_stats(df.loc['Z001M001'])
    plt.figure(figsize=(10, 4))
    for key, series in stats.items():
        plt.plot(series, label=key)
    plt.title("Rolling stats (7d)")
    plt.legend()
    plt.show()


    # Sensor similarity matrix
    features = extract_catch22_features(X_batches)
    sim_matrix = compute_sensor_similarity(features)
    sns.heatmap(sim_matrix, cmap='viridis')
    plt.title("Sensor Similarity (Catch22 + Euclidean)")
    plt.show()
    
    # Dataset clustering dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram

    dataset_feats = aggregate_dataset_features(list_of_X_batches)
    Z = linkage(dataset_feats, method='ward')
    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=[f'DS{i}' for i in range(len(dataset_feats))])
    plt.title("Dataset Clustering (Catch22)")
    plt.show()
    """


    

# Attributes
# ----------

# Functions
# ----------
# Time series analysis
# ---------------------
# plot_time_series(data, x, y, title, xlabel, ylabel)
# plot_time_series_decomposition(data, x, y, title, xlabel, ylabel)
# plot_autocorrelation(data, x, y, title, xlabel, ylabel)
# plot_partial_autocorrelation(data, x, y, title, xlabel, ylabel)
# plot_seasonal_decomposition(data, x, y, title, xlabel, ylabel)
# Intra distribution analysis
# --------------------------
# intra_dataset_correlation(data)
# inter_dataset_correlation(data1, data2)
# Sensor_correlation of dataset
# Temporal correlation of dataset

# Spatial analysis
# ----------------
# plot_map(data, x, y, title, xlabel, ylabel)
# plot_cluster_map(data, x, y, title, xlabel, ylabel)


def plot_time_series(data, start, end, title):
    """
    Plot time series data
    """
    print("Function not implemented yet")
    return data[start:end].plot(title=title)

def plot_correlation_matrix(corr_matrix, title, filename):
    """
    Plot correlation matrix
    """  
    
    corr_matrix.replace([np.inf, -np.inf], 0, inplace=True)
    corr_matrix.fillna(0, inplace=True)
    
    dist_matrix = 1 - np.abs(corr_matrix)    
    # print min and max values of the distance matrix
    print(f"Min value: {np.min(dist_matrix)}")
    print(f"Max value: {np.max(dist_matrix)}")
    # Perform hierarchical clustering using the distance matrix
    linkage_matrix = hierarchy.linkage(dist_matrix, method='average')

    # Get the optimal order for rows and columns
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)  # no_plot to avoid plotting, just to get order
    order = dendro['leaves']

    # Reorder the correlation matrix
    reordered_corr = corr_matrix.iloc[order, order]
    
    f = plt.figure(figsize=(19, 10))
    # Plot the coorelation matrix as square
    plt.matshow(reordered_corr)
    # add colorbar and make sure it is horizontally aligned with the matrix
    cb = plt.colorbar()
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()
    plt.gca().set_aspect('auto')
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    plt.savefig(f"data/correlation_plots/{filename}")
    
    
    return

def check_correlation(data, mode):
    """
    Check correlation of data
    """
    overall_corr = data.corr(min_periods=1)    
    
    plot_correlation_matrix(overall_corr, f"Overall correlation matrix ({mode})", f"{mode}_overall_corr.png")
    return

def check_multistep_correlation(df, mode):
    """
    Check correlation of data
    """
    filtered_columns = [col for col in df.columns if col.weekday() < 5 and 7 <= col.hour < 11]

    # Select only those filtered columns
    filtered_df = df[filtered_columns]

    morning_peak = df[(df.index.weekday < 5) & (df.index.hour >= 7) & (df.index.hour < 10)]
    evening_peak = df[(df.index.weekday < 5) & (df.index.hour >= 16) & (df.index.hour < 19)]
    weekends = df[df.index.weekday >= 5]  # Saturday & Sunday
    
    # print first 20 column index for each data
    print(morning_peak.columns[:20])
    print(evening_peak.columns[:20])
    print(weekends.columns[:20])
    
    corr_values = {}

    for start in pd.date_range(df.index.min(), df.index.max(), freq='2M'):
        end = start + pd.DateOffset(months=2)

        for period, data in {'morning': morning_peak, 'evening': evening_peak, 'weekend': weekends}.items():
            window_data = data.loc[start:end]
            if not window_data.empty:
                corr_values[(start, period)] = window_data.corr().values.flatten()
    
    corr_df = pd.DataFrame(corr_values).T  # Rows = (time, period), Columns = sensor correlations
    meta_corr = corr_df.corr()  # Correlation between correlation trends
    
    print(meta_corr)
    plot_correlation_matrix(meta_corr, f"Multistep correlation matrix ({mode})", f"{mode}_multistep_corr.png")
    
    return

def check_temporal_correlation(data):
    """
    Check temporal correlation of data
    """
    return data.corr()

def plot_heatmap_availability(df, title, start=None, end=None):
    # Ensure columns are datetime
    df.columns = pd.to_datetime(df.columns)

    # Convert to 1 (not NaN) and 0 (NaN)
    df_bin = df.notnull().astype(int)

    # Group columns by month and sum non-NaNs (per sensor per month)
    availability_counts = df_bin.T.groupby(df_bin.columns.to_series().dt.to_period("M")).sum().T
    #availability_counts = df_bin.groupby(df.columns.to_series().dt.to_period("M"), axis=1).sum()
    
    # Count how many days exist per month (independent of NaNs)
    days_per_month = df.columns.to_series().dt.to_period("M").value_counts().sort_index()
    days_per_month = pd.DataFrame([days_per_month.values], index=df.index, columns=days_per_month.index)
    # Final availability (0–1)
    availability = availability_counts / days_per_month
    
    # Replace the nan values with 0
    availability = availability.fillna(0)
    
    if start is not None and end is not None:
        # Filter the availability by start and end date
        availability = availability.loc[:, start:end]


    plt.figure(figsize=(20, 20))
    sns.heatmap(availability, cmap='RdYlGn', cbar=True, cbar_kws={'label': 'Availability (%)'}, linewidths=0.1)
    
    plt.xlabel("Month")
    plt.ylabel("Sensor")
    # Y ticks the sensor ID
    plt.yticks(ticks=np.arange(len(availability.index)), labels=availability.index, rotation=0)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    if start is not None and end is not None:
        plt.xlim(0, len(availability.columns))
    
    if start is not None and end is not None:
        plt.title(f"Sensor availability of {title} ({start} to {end})")
        #plt.savefig(f"data/plots/availability_plots/{title}_availability_{start}_{end}.png", bbox_inches='tight')
    else:
        plt.title(f"Sensor availability of {title}")
        #plt.savefig(f"data/plots/availability_plots/{title}_availability.png", bbox_inches='tight')
    
    return availability


def check_min_data_availability(data, availability):
    # If there are less than 80% of the months available, drop the sensor
    idx_keep = availability.mean(axis=1) > 0.9
    # rows in data that have a true in idx_keep
    data_filter = data.loc[idx_keep, :]
    
    # Get the index of the sensors that are deleted
    idx_delete = availability.index[~idx_keep]
    
    return data_filter, idx_delete

def plot_sensor_availability(data, title_list, start=None, end=None):
    """This functions plots the availability of each sensor over time. The availability is computed per sensor and month.
    The availability is the percentage of non-Nan values for each sensor and month. It is then plotted as heatmap with each sensor as a row and each month as a column.
    Availability under 70% is marked red. For values above there is a fade to green.

    Args:
        data (list of pd df): list of pandas df, with each df containing the data for multiple sensors
    """

    availability_list = []
    for i in range(len(data)):
        df = data[i]
        title = title_list[i]
        availability = plot_heatmap_availability(df, title, start, end)
        availability_list.append(availability)
    
    return availability_list


def drop_useless_sensors(data, title_list, start=None, end=None):
    index_deleted = []
    for i in range(len(data)):
        df = data[i]
        title = title_list[i]
        
        print(f"Checking data availability for {title} - shape before filtering: {df.shape}")
        
        # filter dates
        if start is not None and end is not None:
            df.columns = pd.to_datetime(df.columns)
            start = pd.to_datetime(start)
            end = pd.to_datetime(end) # Get the last date of the month end
            endd = end + pd.DateOffset(months=1) - pd.DateOffset(minutes=1, seconds=1)  # Set to the last second of the month
            
            print(f"Filtering data from {start} to {endd}")
            
            df = df.loc[:, (df.columns >= start) & (df.columns <= endd)]
            print(f"Data shape after filtering by date (from {start} to {endd}): {df.shape}")
            # print column min and max after filtering
            print(f"Data columns after filtering: {df.columns.min()} to {df.columns.max()}")
        
        
        availability = plot_heatmap_availability(df, title, start, end)
        
        data_filter, idx_delete = check_min_data_availability(df, availability)
        #ii = plot_heatmap_availability(data_filter, f"{title}_filtered", start, end)
        data[i] = data_filter
        index_deleted.append(idx_delete)
        
        # print the shape of the df in data
        print(f"Data shape after filtering: {data[i].shape}")
        
        """
        # Also filter the dates of the df, columns in format 2018-01-01 00:00:00, start and end are in format 'YYYY-MM'
        if start is not None and end is not None:
            data[i].columns = pd.to_datetime(data[i].columns)
            start = pd.to_datetime(start)
            end = pd.to_datetime(end) # Get the last date of the month end
            endd = end + pd.DateOffset(months=1) - pd.DateOffset(minutes=1, seconds=1)  # Set to the last second of the month
            
            print(f"Filtering data from {start} to {endd}")
            
            data[i] = data[i].loc[:, (data[i].columns >= start) & (data[i].columns <= endd)]
        """

                
    return data, index_deleted