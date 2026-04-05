"""
Module Order Ablation Study - Results Visualization
Reads results from results_server/results.json and creates comparison plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_file: str = "results_server/results.json") -> Dict:
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def extract_module_order_results(results: Dict) -> pd.DataFrame:
    """
    Extract module order experiments from results
    Returns DataFrame with columns: model_name, mode, order, MAE, MSE, train_time, etc.
    """
    rows = []
    
    for model_name, res in results.items():
        # Check if this is a module order experiment
        # Could be identified by: (1) module_order field, (2) order in config name, or (3) "order_" in model_name
        is_order_exp = False
        module_order = None
                
        # Method 2: Check model name for "order_" pattern
        if "order_" in model_name or "_order_" in model_name:
            is_order_exp = True
            # Extract order from name (e.g., "SpatialFM_bus_order_past_graph_future")
            parts = model_name.split("order_")
            if len(parts) > 1:
                module_order = parts[-1]
        
        # Method 3: Check for specific order names in covariate_config
        order_names = ["past_graph_future", "graph_past_future", "past_future_graph", "graph_only", "raw_fm"]
        config_name = res.get('covariate_config', '')
        if config_name in order_names:
            is_order_exp = True
            module_order = config_name
            print(module_order)
        
        """
        if "raw_fm" in module_order:
            raw_fm_flag = True
            is_order_exp = True
            module_order = f"{module_order.split('_')[0]}_raw_fm"
        """
        
        if is_order_exp and module_order:
            row = {
                'model_name': model_name,
                'mode': res.get('mode', 'unknown'),
                'order': module_order,
                'MAE_masked': res['metrics']['MAE_masked'],
                'MSE_masked': res['metrics']['MSE_masked'],
                'MAE_overall': res['metrics'].get('MAE_overall', np.nan),
                'MSE_overall': res['metrics'].get('MSE_overall', np.nan),
                'train_time': res.get('train_time', 0),
                'inference_time': res.get('inference_time', 0),
                'n_params': res.get('number_of_parameters', 0),
                'loss_history': res.get('loss_history', [])
            }
            rows.append(row)

    # remove duplicates for {mode}_raw_fm (keep only one)
    seen = set()
    unique_rows = []
    for row in rows:
        if row['order'].endswith('raw_fm'):
            if row['mode'] not in seen:
                unique_rows.append(row)
                seen.add(row['mode'])
        else:
            unique_rows.append(row)
    rows = unique_rows
    # 
    
    if not rows:
        print("WARNING: No module order experiments found in results!")
        print("Make sure model names contain 'order_' or covariates_used has module_order field")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    return df


def plot_mae_comparison(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Bar chart comparing MAE across orderings"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Get unique modes
    modes = df['mode'].unique()
    
    for mode in modes:
        mode_df = df[df['mode'] == mode].copy()
        mode_df = mode_df.sort_values('MAE_masked')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        orders = mode_df['order'].values
        mae_values = mode_df['MAE_masked'].values
        
        # Color bars: best=green, worst=red, others=blue
        colors = ['#2ecc71' if i == 0 else '#e74c3c' if i == len(orders)-1 else '#3498db' 
                  for i in range(len(orders))]
        
        bars = ax.bar(range(len(orders)), mae_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mae_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Calculate improvement percentages
        baseline_mae = mae_values[-1]  # worst
        improvements = [(baseline_mae - mae) / baseline_mae * 100 for mae in mae_values]
        
        # Add improvement percentage as secondary text
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                   f'+{imp:.1f}%' if imp >= 0 else f'{imp:.1f}%',
                   ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
        
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels(orders, rotation=45, ha='right')
        ax.set_ylabel('MAE (masked)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Module Ordering', fontsize=12, fontweight='bold')
        ax.set_title(f'Module Order Comparison - {mode.upper()}\n(Lower is better)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Best'),
            Patch(facecolor='#3498db', label='Intermediate'),
            Patch(facecolor='#e74c3c', label='Worst')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mae_comparison_{mode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_dir}/mae_comparison_{mode}.png")


def plot_training_curves(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Plot training loss curves for different orderings"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    modes = df['mode'].unique()
    
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for idx, (_, row) in enumerate(mode_df.iterrows()):
            order = row['order']
            loss_history = row['loss_history']
            
            if loss_history and len(loss_history) > 0:
                color = colors[idx % len(colors)]
                ax.plot(loss_history, label=order, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Loss (MAE)', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Curves - {mode.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves_{mode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_dir}/training_curves_{mode}.png")


def plot_heatmap_mode_order(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Heatmap showing MAE for each mode × order combination"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Pivot table: modes as rows, orders as columns
    pivot_df = df.pivot_table(
        values='MAE_masked',
        index='mode',
        columns='order',
        aggfunc='mean'
    )
    
    if pivot_df.empty:
        print("Not enough data for heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'MAE (masked)'}, linewidths=0.5,
                ax=ax, vmin=pivot_df.min().min(), vmax=pivot_df.max().max())
    
    ax.set_title('Module Order Performance Heatmap\n(Darker green = better)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Module Ordering', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transport Mode', fontsize=12, fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmap_mode_order.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/heatmap_mode_order.png")


def plot_improvement_percentage(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Plot improvement percentage relative to baseline (worst ordering)"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    modes = df['mode'].unique()
    
    fig, axes = plt.subplots(1, len(modes), figsize=(5*len(modes), 5))
    if len(modes) == 1:
        axes = [axes]
    
    for idx, mode in enumerate(modes):
        mode_df = df[df['mode'] == mode].copy()
        mode_df = mode_df.sort_values('MAE_masked')
        
        # Calculate improvement from worst
        baseline_mae = mode_df['MAE_masked'].max()
        mode_df['improvement'] = (baseline_mae - mode_df['MAE_masked']) / baseline_mae * 100
        
        ax = axes[idx]
        
        # Bar chart
        orders = mode_df['order'].values
        improvements = mode_df['improvement'].values
        
        colors = ['#2ecc71' if imp > 5 else '#f39c12' if imp > 2 else '#e74c3c' 
                  for imp in improvements]
        
        bars = ax.barh(range(len(orders)), improvements, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {val:.1f}%',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(orders)))
        ax.set_yticklabels(orders)
        ax.set_xlabel('Improvement over Worst (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{mode.upper()}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add significance markers
        ax.axvline(x=2, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Minor (2%)')
        ax.axvline(x=5, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Major (5%)')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Improvement Relative to Worst Ordering', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/improvement_percentage.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/improvement_percentage.png")


def plot_time_vs_performance(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Scatter plot: training time vs performance"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = df['mode'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(modes)))
    
    for mode, color in zip(modes, colors):
        mode_df = df[df['mode'] == mode]
        
        ax.scatter(mode_df['train_time'], mode_df['MAE_masked'], 
                  s=100, alpha=0.7, color=color, edgecolors='black', linewidth=1.5,
                  label=mode)
        
        # Add order labels
        for _, row in mode_df.iterrows():
            ax.annotate(row['order'], 
                       (row['train_time'], row['MAE_masked']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (masked)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time vs Performance\n(All orderings have similar compute cost)', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Mode', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_vs_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/time_vs_performance.png")


def plot_fm_order_table(results_file: str = "results_server/results.json",
                        save_dir: str = "results_server/plots"):
    """
    One table plot per mode.
    Columns = FM types, Rows = module order types.
    Cell value = min MAE_masked across all exo-variable configs.
    Best per column: bold. Second best per column: underlined (via mathtext).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(results_file) as f:
        results = json.load(f)

    ORDER_TYPES = ['past_future_graph', 'graph_past_future', 'past_graph_future', 'graph_only', 'cov_only']
    ORDER_LABELS = {
        'past_future_graph': 'Past→Future→Graph',
        'graph_past_future': 'Graph→Past→Future',
        'past_graph_future': 'Past→Graph→Future',
        'graph_only':        'Graph Only',
        'cov_only':          'Cov Only',
    }

    rows = []
    for model_name, res in results.items():
        if '_order_' not in model_name:
            continue
        after = model_name.split('_order_', 1)[1]   # {mode}_{FM}_{order_type}_{exo}
        for ot in ORDER_TYPES:
            if ot in after:
                idx = after.index(ot)
                pre_parts = after[:idx].rstrip('_').split('_')
                mode = pre_parts[0]
                fm = '_'.join(pre_parts[1:]) if len(pre_parts) > 1 else 'none'
                rows.append({
                    'mode': mode,
                    'fm': fm,
                    'order_type': ot,
                    'MAE': res['metrics']['MAE_masked'],
                })
                break

    if not rows:
        print("No data found for fm_order_table")
        return

    df = pd.DataFrame(rows)

    # Best MAE per (mode, fm, order_type) across all exo configs
    grouped = df.groupby(['mode', 'fm', 'order_type'])['MAE'].min().reset_index()

    mode_order = sorted(grouped['mode'].unique())

    for mode in mode_order:
        mode_df = grouped[grouped['mode'] == mode]

        pivot = mode_df.pivot_table(values='MAE', index='order_type', columns='fm', aggfunc='min')

        # Preserve canonical row order
        row_order = [ot for ot in ORDER_TYPES if ot in pivot.index]
        pivot = pivot.loc[row_order]

        n_rows, n_cols = pivot.shape
        col_labels = list(pivot.columns)
        row_labels = [ORDER_LABELS.get(ot, ot) for ot in row_order]

        # Identify best (rank=1) and second-best (rank=2) per column
        rank_map = {}
        for ci, _ in enumerate(col_labels):
            col_vals = [(ri, pivot.iloc[ri, ci])
                        for ri in range(n_rows)
                        if not np.isnan(pivot.iloc[ri, ci])]
            col_vals.sort(key=lambda x: x[1])
            if len(col_vals) >= 1:
                rank_map[(col_vals[0][0], ci)] = 1
            if len(col_vals) >= 2:
                rank_map[(col_vals[1][0], ci)] = 2

        # Build plain cell text (formatting applied via table cell properties)
        cell_text = []
        cell_colors = []
        for ri in range(n_rows):
            row_text, row_colors = [], []
            for ci in range(n_cols):
                val = pivot.iloc[ri, ci]
                if np.isnan(val):
                    row_text.append('—')
                    row_colors.append('#f0f0f0')
                else:
                    row_text.append(f'{val:.3f}')
                    rank = rank_map.get((ri, ci), 0)
                    if rank == 1:
                        row_colors.append('#d5f5e3')   # light green for best
                    elif rank == 2:
                        row_colors.append('#fef9e7')   # light yellow for second
                    else:
                        row_colors.append('white')
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        fig_w = max(6, 1.8 * n_cols + 2.5)
        fig_h = max(3, 0.7 * n_rows + 1.8)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            rowLoc='right',
            loc='center',
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 2.2)

        # Apply bold / underline via text properties after table creation
        for ri in range(n_rows):
            for ci in range(n_cols):
                rank = rank_map.get((ri, ci), 0)
                txt = table[(ri + 1, ci)].get_text()
                if rank == 1:
                    txt.set_fontweight('bold')
                elif rank == 2:
                    # Simulate underline: bold-italic + underline via annotation trick
                    txt.set_fontstyle('italic')
                    txt.set_fontweight('bold')
                    # Draw an underline segment after rendering
                    table[(ri + 1, ci)]._text = txt

        # Header row style
        for ci in range(n_cols):
            cell = table[(0, ci)]
            cell.set_facecolor('#2c3e50')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')

        # Row-label column style
        for ri in range(1, n_rows + 1):
            cell = table[(ri, -1)]
            cell.set_facecolor('#34495e')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
            cell.get_text().set_fontsize(9)

        ax.set_title(
            f'MAE (masked) — {mode.upper()}\n'
            'bold+green = best per FM column   |   bold-italic+yellow = second best',
            fontsize=12, fontweight='bold', pad=16
        )

        plt.tight_layout()
        out_path = f"{save_dir}/fm_order_table_{mode}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {out_path}")


def generate_summary_table(df: pd.DataFrame, save_dir: str = "results_server/plots"):
    """Generate summary table and save as image"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Create summary statistics
    summary_rows = []
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].sort_values('MAE_masked')
        
        best = mode_df.iloc[0]
        worst = mode_df.iloc[-1]
        
        improvement = (worst['MAE_masked'] - best['MAE_masked']) / worst['MAE_masked'] * 100
        
        summary_rows.append({
            'Mode': mode.upper(),
            'Best Order': best['order'],
            'Best MAE': f"{best['MAE_masked']:.4f}",
            'Worst Order': worst['order'],
            'Worst MAE': f"{worst['MAE_masked']:.4f}",
            'Improvement': f"{improvement:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 2 + 0.5*len(summary_rows)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#3498db']*len(summary_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Module Order Ablation Study - Summary', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f"{save_dir}/summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/summary_table.png")
    
    return summary_df


def print_text_summary(df: pd.DataFrame):
    """Print text summary to console"""
    if df.empty:
        print("No data to summarize")
        return
    
    print("\n" + "="*80)
    print("MODULE ORDER ABLATION STUDY - RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].sort_values('MAE_masked')
        
        table_rows = []
        
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        
        baseline_mae = mode_df['MAE_masked'].max()
        # get raw_fm baseline if exists
        raw_fm_baseline = mode_df[mode_df['order'].str.contains('raw_fm')]['MAE_masked'].max()
        if not pd.isna(raw_fm_baseline):
            baseline_mae = raw_fm_baseline

        for _, row in mode_df.iterrows():
            improvement = (baseline_mae - row['MAE_masked']) / baseline_mae * 100
            table_rows.append((row['order'], row['MAE_masked'], row['MSE_masked'], row['train_time'], improvement))

        
        best = mode_df.iloc[0]
        worst = mode_df.iloc[-1]
        improvement = (worst['MAE_masked'] - best['MAE_masked']) / worst['MAE_masked'] * 100
        
        print(f"{'-'*80}")
        print(f"Best: {best['order']} (MAE: {best['MAE_masked']:.4f})")
        print(f"Worst: {worst['order']} (MAE: {worst['MAE_masked']:.4f})")
        print(f"Improvement: {improvement:.2f}%")
            
        table_df = pd.DataFrame(table_rows, columns=['Order', 'MAE', 'MSE', 'Train Time', 'Improvement'])
        
        table_df = reshape_tex_tables(table_df)
        
        latex_table = table_df.to_latex(index=False, caption=f'Module Order Results - {mode}',
                                        label=f'tab:module_order_{mode.lower()}', float_format="%.4f")
        with open(f"results_server/plots/text_summary_{mode.lower()}.tex", 'w') as f:
            f.write(latex_table)
        
def _df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Build a LaTeX table string from a DataFrame without requiring jinja2."""
    cols = list(df.columns)
    col_fmt = "l" * len(cols)
    lines = []
    lines.append("\\begin{table*}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def _load_and_filter(results_file: str, results: Dict, filter_date) -> Dict:
    """Load results from file if not provided, then apply optional date filter."""
    if results is None:
        with open(results_file) as f:
            results = json.load(f)
    if filter_date:
        cutoff = pd.to_datetime(filter_date)
        results = {k: v for k, v in results.items()
                   if pd.to_datetime(v.get('creation_time', '2000-01-01')) >= cutoff}
    return results


def generate_fm_text_summaries(results_file: str = "results_server/results.json",
                               save_dir: str = "results_server/plots",
                               results: Dict = None,
                               filter_date=None):
    """Generate one LaTeX text_summary per foundation model (across all modes).

    Args:
        results_file: Path to JSON results file (used only if results is None)
        save_dir: Directory to save .tex files
        results: Pre-loaded (and optionally pre-filtered) results dict. If provided,
                 results_file is ignored, so date filtering applied upstream is respected.
        filter_date: Optional date string; entries created before this date are excluded.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = _load_and_filter(results_file, results, filter_date)

    ORDER_TYPES = ['past_future_graph', 'graph_past_future', 'past_graph_future', 'graph_only', 'cov_only']

    rows = []
    for model_name, res in results.items():
        if '_order_' not in model_name:
            continue
        after = model_name.split('_order_', 1)[1]

        # Try ORDER_TYPES first (handles entries like {mode}_{fm}_{order_type}_raw_fm correctly)
        matched = False
        for ot in ORDER_TYPES:
            if ot in after:
                idx = after.index(ot)
                pre_parts = after[:idx].rstrip('_').split('_')
                mode = pre_parts[0]
                fm = '_'.join(pre_parts[1:]) if len(pre_parts) > 1 else 'none'
                exo = after[idx + len(ot):].lstrip('_')
                order_str = f"{ot}_{exo}" if exo else ot
                rows.append({
                    'mode': mode,
                    'fm': fm,
                    'Order': order_str,
                    'MAE': res['metrics']['MAE_masked'],
                    'MSE': res['metrics']['MSE_masked'],
                    'Train Time': res.get('train_time', 0),
                })
                matched = True
                break

        # Fall back: pure raw_fm run (order_type=raw_fm, no ORDER_TYPES token present)
        if not matched and 'raw_fm' in after:
            pre = after.replace('_raw_fm', '').rstrip('_')
            pre_parts = pre.split('_')
            mode = pre_parts[0]
            fm = '_'.join(pre_parts[1:]) if len(pre_parts) > 1 else 'none'
            rows.append({
                'mode': mode,
                'fm': fm,
                'Order': 'raw_fm',
                'MAE': res['metrics']['MAE_masked'],
                'MSE': res['metrics']['MSE_masked'],
                'Train Time': res.get('train_time', 0),
            })

    if not rows:
        print("No data found for FM text summaries")
        return

    df = pd.DataFrame(rows)
    
    print(df['fm'].unique())

    for fm in sorted(df['fm'].unique()):
        if not fm:
            continue
        fm_df = df[df['fm'] == fm].copy()
        
        print(df['mode'].unique())

        for mode in sorted(fm_df['mode'].unique()):
            mode_df = fm_df[fm_df['mode'] == mode].sort_values('MAE').reset_index(drop=True)

            # Baseline: raw_fm row for this mode/FM combo, else worst MAE
            raw_fm_rows = mode_df[mode_df['Order'].str.contains('raw_fm')]
            baseline_mae = raw_fm_rows['MAE'].max() if not raw_fm_rows.empty else mode_df['MAE'].max()

            mode_df = mode_df.copy()
            mode_df['Improvement'] = (baseline_mae - mode_df['MAE']) / baseline_mae * 100

            tdf = mode_df[['Order', 'MAE', 'MSE', 'Train Time', 'Improvement']].copy()
            shaped = reshape_tex_tables(tdf)

            latex_table = _df_to_latex(
                shaped,
                caption=f'Module Order Results - {fm} / {mode}',
                label=f'tab:module_order_{fm.lower()}_{mode.lower()}',
            )
            out_path = f"{save_dir}/text_summary_fm_{fm.lower()}_{mode.lower()}.tex"
            with open(out_path, 'w') as fh:
                fh.write(latex_table)
            print(f"✓ Saved: {out_path}")


def reshape_tex_tables(df):
    
    # split the name into multiple columns with x for option
    # columns:
    # - order: past_graph_future (pgf) graph_past_future (gpf), past_future_graph (pfg), graph_only (g), raw_fm (fm)
    # - cov: weather (wea), events (evt), date
    df['pgf'] = df['Order'].apply(lambda x: "x" if 'past_graph_future' in x else "")
    df['gpf'] = df['Order'].apply(lambda x: "x" if 'graph_past_future' in x else "")
    df['pfg'] = df['Order'].apply(lambda x: "x" if 'past_future_graph' in x else "")
    df['g']   = df['Order'].apply(lambda x: "x" if 'graph_only' in x else "")
    df['co']  = df['Order'].apply(lambda x: "x" if 'cov_only' in x else "")
    df['fm']  = df['Order'].apply(lambda x: "x" if 'raw_fm' in x else "")
    
    df['wea']  = df['Order'].apply(lambda x: "x" if 'weather' in x or 'all_cov' in x else "")
    df['evt']  = df['Order'].apply(lambda x: "x" if 'event' in x or 'all_cov' in x else "")
    df['date'] = df['Order'].apply(lambda x: "x" if 'date' in x or 'all_cov' in x else "")
    
    # reorder columns
    cols = ['pgf', 'gpf', 'pfg', 'g', 'co', 'fm', 'wea', 'evt', 'date', 'MAE', 'MSE', 'Train Time', 'Improvement']
    df = df[cols]    
    
    # all number to 2 decimal places
    df['MAE'] = df['MAE'].apply(lambda x: f"{x:.4f}")
    df['MSE'] = df['MSE'].apply(lambda x: f"{x:.4f}")
    df['Train Time'] = df['Train Time'].apply(lambda x: f"{x:.0f}")
    df['Improvement'] = df['Improvement'].apply(lambda x: f"{x:.2f}\%")
    
    return df


def _order_sort_key(order_str: str) -> Tuple[int, int]:
    """Return (order_rank, cov_rank) for canonical row sorting."""
    ORDER_RANK = {'raw_fm': 0, 'graph_only': 1, 'cov_only': 2, 'graph_past_future': 3,
                  'past_future_graph': 4, 'past_graph_future': 5}
    order_rank = next((v for k, v in ORDER_RANK.items() if k in order_str), 5)
    # cov_rank: encode (date, weather, event) as 3-bit integer for stable ordering
    cov_rank = (('date' in order_str or 'all_cov' in order_str) * 4 +
                ('weather' in order_str or 'all_cov' in order_str) * 2 +
                ('event' in order_str or 'all_cov' in order_str) * 1)
    return (order_rank, cov_rank)


def generate_mode_wide_tables(results_file: str = "results_server/results.json",
                              save_dir: str = "results_server/plots",
                              results: Dict = None,
                              filter_date=None,
                              show_std: bool = False):
    """One LaTeX table per mode.

    Rows    = model configurations (order + covariate flags).
    Columns = one group per FM, each group has MAE / MSE / Improvement vs raw_fm.

    Args:
        results_file: Path to JSON results file (used only if results is None)
        save_dir: Directory to save .tex files
        results: Pre-loaded (and optionally pre-filtered) results dict.
        filter_date: Optional date string; entries created before this date are excluded.
        show_std: If True, render MAE/RMSE as ``mean ± std`` (2 dp) using metrics_std
                  where available, and save to wide_table_{mode}_std.tex.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = _load_and_filter(results_file, results, filter_date)

    ORDER_TYPES = ['past_future_graph', 'graph_past_future', 'past_graph_future', 'graph_only', 'cov_only']

    # ── parse rows (same logic as generate_fm_text_summaries) ──────────────
    rows = []
    for model_name, res in results.items():
        if '_order_' not in model_name:
            continue
        after = model_name.split('_order_', 1)[1]
        matched = False
        for ot in ORDER_TYPES:
            if ot in after:
                idx = after.index(ot)
                pre_parts = after[:idx].rstrip('_').split('_')
                mode = pre_parts[0]
                fm = '_'.join(pre_parts[1:]) if len(pre_parts) > 1 else 'none'
                exo = after[idx + len(ot):].lstrip('_')
                order_str = f"{ot}_{exo}" if exo else ot
                std = res.get('metrics_std') or {}
                rows.append({'mode': mode, 'fm': fm, 'Order': order_str,
                             'MAE': res['metrics']['MAE_masked'],
                             'MSE': res['metrics']['MSE_masked'],
                             'MAE_std': std.get('MAE_masked'),
                             'MSE_std': std.get('MSE_masked')})
                matched = True
                break
        if not matched and 'raw_fm' in after:
            pre = after.replace('_raw_fm', '').rstrip('_')
            pre_parts = pre.split('_')
            mode = pre_parts[0]
            fm = '_'.join(pre_parts[1:]) if len(pre_parts) > 1 else 'none'
            std = res.get('metrics_std') or {}
            rows.append({'mode': mode, 'fm': fm, 'Order': 'raw_fm',
                         'MAE': res['metrics']['MAE_masked'],
                         'MSE': res['metrics']['MSE_masked'],
                         'MAE_std': std.get('MAE_masked'),
                         'MSE_std': std.get('MSE_masked')})

    if not rows:
        print("No data found for mode wide tables")
        return

    df = pd.DataFrame(rows)
    df = df[df['fm'] != 'none']          # drop old-format entries without FM name

    # ── derive display flags ────────────────────────────────────────────────
    def _flags(order):
        has_graph = any(ot in order for ot in
                        ['graph_only', 'past_graph_future', 'graph_past_future', 'past_future_graph'])
        has_cov = any(ot in order for ot in
                      ['cov_only', 'past_graph_future', 'graph_past_future', 'past_future_graph'])
        if 'past_graph_future' in order:
            order_label = 'pgf'
        elif 'graph_past_future' in order:
            order_label = 'gpf'
        elif 'past_future_graph' in order:
            order_label = 'pfg'
        else:
            order_label = ''
        return {
            'fm_flag': 'x',
            'g':       'x' if has_graph else '',
            'cov':     'x' if has_cov   else '',
            'order':   order_label,
        }

    flag_df = df['Order'].apply(lambda o: pd.Series(_flags(o)))
    df = pd.concat([df, flag_df], axis=1)

    fms = [f for f in ['chronos', 'flowstate', 'moment', 'timesfm25']
           if f in df['fm'].values]

    for mode in sorted(df['mode'].unique()):
        mode_df = df[df['mode'] == mode].copy()

        # raw_fm baseline per FM
        baselines = {}
        for fm in fms:
            base_rows = mode_df[(mode_df['fm'] == fm) & (mode_df['Order'] == 'raw_fm')]
            if not base_rows.empty:
                baselines[fm] = base_rows['MAE'].iloc[0]

        # canonical set of Orders (rows), sorted
        all_orders = sorted(mode_df['Order'].unique(), key=_order_sort_key)

        # ── build LaTeX ─────────────────────────────────────────────────────
        FLAG_COLS        = ['fm_flag', 'g', 'cov', 'order']
        FLAG_COLS_LABELS = ['fm',     'g', 'cov', 'order']
        n_flag = len(FLAG_COLS)

        lines = []
        lines.append(r'\begin{table*}')
        lines.append(r'\centering')
        lines.append(r'\tiny')
        lines.append(rf'\caption{{Module Order Results --- {mode.upper()} (MAE / RMSE / \% vs raw FM)}}')
        lines.append(rf'\label{{tab:wide_{mode.lower()}}}')

        # column spec: flags left-aligned, then for each FM: l l r (MAE/RMSE left for ±, Impr right)
        fm_cols = ' l l r' if show_std else ' r r r'
        col_spec = 'l' * n_flag + (''.join(fm_cols for _ in fms))
        lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
        lines.append(r'\toprule')

        # header row 1: spanning FM names
        header1_parts = [r'\multicolumn{' + str(n_flag) + r'}{c}{}']
        for fm in fms:
            header1_parts.append(rf'\multicolumn{{3}}{{c}}{{\textbf{{{fm}}}}}')
        lines.append(' & '.join(header1_parts) + r' \\')
        # cmidrules under each FM group
        for i, _ in enumerate(fms):
            start = n_flag + i * 3 + 1
            end   = start + 2
            lines[-1] += rf' \cmidrule(lr){{{start}-{end}}}'
        # that appended in-line — move to a separate line
        lines[-1] = lines[-1]  # already done above

        # header row 2: flag names + MAE/MSE/Impr per FM
        header2_parts = FLAG_COLS_LABELS[:]
        for _ in fms:
            header2_parts += ['MAE', 'RMSE', 'Impr']
        lines.append(' & '.join(header2_parts) + r' \\')
        lines.append(r'\midrule')

        def _fmt_metric(val, std_val):
            """Format a metric cell: 'X.XX ± Y.YY' or 'X.XX[XX]' depending on show_std."""
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return '—'
            if show_std:
                if std_val is not None and not (isinstance(std_val, float) and np.isnan(std_val)):
                    return rf'{val:.2f} $\pm$ {std_val:.2f}'
                return f'{val:.2f}'
            return f'{val:.4f}'

        # data rows
        for order_str in all_orders:
            order_rows = mode_df[mode_df['Order'] == order_str]
            if order_rows.empty:
                continue
            first = order_rows.iloc[0]
            row_parts = [first[c] for c in FLAG_COLS]

            for fm in fms:
                fm_rows = order_rows[order_rows['fm'] == fm]
                if fm_rows.empty:
                    row_parts += ['—', '—', '—']
                else:
                    r = fm_rows.iloc[0]
                    mae = r['MAE']
                    mse = r['MSE']
                    mae_std = r.get('MAE_std') if hasattr(r, 'get') else r['MAE_std'] if 'MAE_std' in r.index else None
                    mse_std = r.get('MSE_std') if hasattr(r, 'get') else r['MSE_std'] if 'MSE_std' in r.index else None
                    # pandas Series: use direct access
                    try:
                        mae_std = r['MAE_std'] if not pd.isna(r['MAE_std']) else None
                        mse_std = r['MSE_std'] if not pd.isna(r['MSE_std']) else None
                    except (KeyError, TypeError):
                        mae_std, mse_std = None, None
                    baseline = baselines.get(fm)
                    if (baseline and baseline > 0
                            and mae is not None
                            and not (isinstance(mae, float) and np.isnan(mae))):
                        impr = (baseline - mae) / baseline * 100
                        impr_str = rf'{impr:+.1f}\%'
                    else:
                        impr_str = '—'
                    row_parts += [_fmt_metric(mae, mae_std), _fmt_metric(mse, mse_std), impr_str]

            lines.append(' & '.join(str(p) for p in row_parts) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table*}')
        latex = '\n'.join(lines) + '\n'

        suffix = '_std' if show_std else ''
        out_path = f"{save_dir}/wide_table_{mode.lower()}{suffix}.tex"
        with open(out_path, 'w') as fh:
            fh.write(latex)
        print(f"✓ Saved: {out_path}")


def generate_benchmark_table(results_file: str = "results_server/results.json",
                             save_dir: str = "results_server/plots",
                             results: Dict = None,
                             filter_date=None):
    """LaTeX table: reference models vs TimesFM (raw) vs TimesFM+Spatial (best per mode).

    Rows    = models (ref baselines, then FM rows).
    Columns = one group per transport mode, each group has MAE and RMSE sub-columns.
    Formatting: bold = best per column, underline = second-best per column.

    Args:
        results_file: Path to JSON (used for ref models, always unfiltered).
        save_dir:     Output directory.
        results:      Pre-filtered FM results dict (for FM rows). If None, reads file.
        filter_date:  Applied to FM rows when results is None.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Ref models always loaded from full file (no date filter — they are the baselines)
    with open(results_file) as f:
        all_unfiltered = json.load(f)

    # FM results may be date-filtered
    fm_results = _load_and_filter(results_file, results, filter_date)

    MODES       = ["bike", "ped", "road"]
    MODE_LABELS = {"bike": "Bike", "ped": "Pedestrian", "road": "Road"}

    REF_ROWS = [
        ("ref_staef",    "STAEformer"),
        ("ref_stgformer","STGformer"),
        ("ref_exost",    "ExoST"),
        ("ref_tftexost", "TFTExoST"),
    ]

    def _get_mae_rmse(entry):
        """Return (mae, rmse) handling both ref and FM metric key names."""
        m = entry["metrics"]
        mae  = m.get("MAE_masked")
        rmse = m.get("RMSE_masked") or m.get("MSE_masked")  # ref uses RMSE_masked, FM uses MSE_masked (which is RMSE)
        return (mae, rmse) if (mae is not None and rmse is not None) else None

    # ── collect rows ─────────────────────────────────────────────────────────
    # Each row: (label, {mode: (mae, rmse)}, is_separator_above)
    table_rows = []

    for key_prefix, label in REF_ROWS:
        mode_data = {}
        for mode in MODES:
            entry = all_unfiltered.get(f"{key_prefix}_{mode}")
            if entry:
                vals = _get_mae_rmse(entry)
                if vals:
                    mode_data[mode] = vals
        table_rows.append((label, mode_data, False))

    # separator before FM rows
    # TimesFM raw (deterministic baseline)
    tfm_raw = {}
    for mode in MODES:
        entry = fm_results.get(f"SpatialFM_order_{mode}_timesfm25_raw_fm_raw_fm")
        if entry:
            vals = _get_mae_rmse(entry)
            if vals:
                tfm_raw[mode] = vals
    table_rows.append(("TimesFM (raw)", tfm_raw, True))  # True = midrule above

    # TimesFM + Spatial: best (min MAE) non-raw config per mode
    tfm_best = {}
    tfm_best_label = {}
    for mode in MODES:
        candidates = {k: v for k, v in fm_results.items()
                      if f"_order_{mode}_timesfm25_" in k and "raw_fm" not in k}
        if candidates:
            best_key = min(candidates, key=lambda k: candidates[k]["metrics"]["MAE_masked"])
            vals = _get_mae_rmse(candidates[best_key])
            if vals:
                tfm_best[mode] = vals
                # extract order+cov label from key for caption note
                tfm_best_label[mode] = best_key.split(f"_order_{mode}_timesfm25_")[1]
    best_configs_note = "; ".join(
        f"{m}: {tfm_best_label.get(m,'—').replace('_', ' ')}" for m in MODES if m in tfm_best_label
    )
    table_rows.append((r"TimesFM + Spatial\textsuperscript{$\dagger$}", tfm_best, False))

    # ── find best / second-best per column ───────────────────────────────────
    # columns: (mode, metric_idx) where metric_idx 0=MAE, 1=RMSE
    col_values: Dict[Tuple, List[Tuple[int, float]]] = {}  # col → [(row_idx, val)]
    for ri, (_, mode_data, _) in enumerate(table_rows):
        for mode in MODES:
            if mode in mode_data:
                mae, rmse = mode_data[mode]
                col_values.setdefault((mode, 0), []).append((ri, mae))
                col_values.setdefault((mode, 1), []).append((ri, rmse))

    best_cells: Dict[Tuple[int, str, int], int] = {}  # (row_idx, mode, metric_idx) → rank (1 or 2)
    for col_key, vals in col_values.items():
        mode, mi = col_key
        sorted_vals = sorted(vals, key=lambda x: x[1])
        if len(sorted_vals) >= 1:
            best_cells[(sorted_vals[0][0], mode, mi)] = 1
        if len(sorted_vals) >= 2:
            best_cells[(sorted_vals[1][0], mode, mi)] = 2

    def _fmt(val, row_idx, mode, mi):
        if val is None:
            return "—"
        s = f"{val:.4f}"
        rank = best_cells.get((row_idx, mode, mi), 0)
        if rank == 1:
            return rf"\textbf{{{s}}}"
        if rank == 2:
            return rf"\underline{{{s}}}"
        return s

    # ── build LaTeX ───────────────────────────────────────────────────────────
    n_modes = len(MODES)
    col_spec = "l" + " rr" * n_modes

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Comparison of reference models and TimesFM variants "
                 r"(MAE / RMSE, lower is better). "
                 rf"\textsuperscript{{$\dagger$}}Best covariate config per mode: {best_configs_note}.}}")
    lines.append(r"\label{tab:benchmark_comparison}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: mode group labels
    h1 = [r"\multicolumn{1}{c}{}"]
    for mode in MODES:
        h1.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{MODE_LABELS[mode]}}}}}")
    lines.append(" & ".join(h1) + r" \\")
    for i, _ in enumerate(MODES):
        start = 2 + i * 2
        lines[-1] += rf" \cmidrule(lr){{{start}-{start+1}}}"

    # Header row 2: Model | MAE RMSE | MAE RMSE | ...
    h2 = ["Model"]
    for _ in MODES:
        h2 += ["MAE", "RMSE"]
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for ri, (label, mode_data, sep_above) in enumerate(table_rows):
        if sep_above:
            lines.append(r"\midrule")
        cells = [label]
        for mode in MODES:
            if mode in mode_data:
                mae, rmse = mode_data[mode]
                cells.append(_fmt(mae,  ri, mode, 0))
                cells.append(_fmt(rmse, ri, mode, 1))
            else:
                cells += ["—", "—"]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    latex = "\n".join(lines) + "\n"

    out_path = f"{save_dir}/benchmark_table.tex"
    with open(out_path, "w") as fh:
        fh.write(latex)
    print(f"✓ Saved: {out_path}")


def generate_benchmark_timing_plots(results_file: str = "results_server/results.json",
                                    save_dir: str = "results_server/plots",
                                    results: Dict = None,
                                    filter_date=None):
    """Scatter plots: MAE vs train_time and MAE vs inference_time, one figure per mode.

    Models shown:
      - Reference baselines: STAEformer, STGformer, ExoST, TFTExoST
      - TimesFM (raw)
      - TimesFM + graph_only
      - TimesFM + cov_only  (if available)
      - TimesFM + best full config (graph + all covariates)

    Produces 6 files: benchmark_timing_train_{mode}.pdf  x3
                      benchmark_timing_infer_{mode}.pdf  x3
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(results_file) as f:
        all_unfiltered = json.load(f)

    fm_results = _load_and_filter(results_file, results, filter_date)

    MODES       = ["bike", "ped", "road"]
    MODE_LABELS = {"bike": "Bike", "ped": "Pedestrian", "road": "Road"}

    # colour / marker scheme
    STYLE = {
        "STAEformer":       {"color": "#4C72B0", "marker": "s",  "zorder": 3},
        "STGformer":        {"color": "#DD8452", "marker": "^",  "zorder": 3},
        "ExoST":            {"color": "#55A868", "marker": "D",  "zorder": 3},
        "TFTExoST":         {"color": "#C44E52", "marker": "P",  "zorder": 3},
        "TimesFM (raw)":    {"color": "#8172B2", "marker": "o",  "zorder": 4},
        "TimesFM + graph":  {"color": "#8172B2", "marker": "o",  "zorder": 4},
        "TimesFM + cov":    {"color": "#8172B2", "marker": "o",  "zorder": 4},
        "TimesFM + best":   {"color": "#8172B2", "marker": "o",  "zorder": 5},
    }
    # line styles to distinguish FM variants (same colour family)
    FM_LINESTYLE = {
        "TimesFM (raw)":    {"linestyle": "none", "edgecolors": "#8172B2", "facecolors": "white",  "s": 90},
        "TimesFM + graph":  {"linestyle": "none", "edgecolors": "#8172B2", "facecolors": "#c4b4e0","s": 90},
        "TimesFM + cov":    {"linestyle": "none", "edgecolors": "#8172B2", "facecolors": "#8172B2","s": 90},
        "TimesFM + best":   {"linestyle": "none", "edgecolors": "#8172B2", "facecolors": "#8172B2","s": 140},
    }

    def _get_mae(entry):
        m = entry["metrics"]
        return m.get("MAE_masked")

    def _get_times(entry):
        return entry.get("train_time", 0), entry.get("inference_time", 0)

    for mode in MODES:
        # ── collect model rows ────────────────────────────────────────────
        rows = []

        # Reference baselines
        for key_prefix, label in [
            ("ref_staef",    "STAEformer"),
            ("ref_stgformer","STGformer"),
            ("ref_exost",    "ExoST"),
            ("ref_tftexost", "TFTExoST"),
        ]:
            entry = all_unfiltered.get(f"{key_prefix}_{mode}")
            if entry:
                mae = _get_mae(entry)
                tr, inf = _get_times(entry)
                if mae is not None:
                    rows.append({"label": label, "mae": mae,
                                 "train_time": tr, "infer_time": inf,
                                 "group": "ref"})

        # TimesFM raw
        entry = fm_results.get(f"SpatialFM_order_{mode}_timesfm25_raw_fm_raw_fm")
        if entry:
            mae = _get_mae(entry)
            tr, inf = _get_times(entry)
            if mae is not None:
                rows.append({"label": "TimesFM (raw)", "mae": mae,
                             "train_time": tr, "infer_time": inf, "group": "fm"})

        # TimesFM + graph_only
        entry = fm_results.get(f"SpatialFM_order_{mode}_timesfm25_graph_only_no_cov")
        if entry:
            mae = _get_mae(entry)
            tr, inf = _get_times(entry)
            if mae is not None:
                rows.append({"label": "TimesFM + graph", "mae": mae,
                             "train_time": tr, "infer_time": inf, "group": "fm"})

        # TimesFM + cov_only
        entry = fm_results.get(f"SpatialFM_order_{mode}_timesfm25_cov_only_cov_only")
        if entry:
            mae = _get_mae(entry)
            tr, inf = _get_times(entry)
            if mae is not None:
                rows.append({"label": "TimesFM + cov", "mae": mae,
                             "train_time": tr, "infer_time": inf, "group": "fm"})

        # TimesFM + best full config (min MAE among non-raw, graph+cov configs)
        candidates = {k: v for k, v in fm_results.items()
                      if f"_order_{mode}_timesfm25_" in k
                      and "raw_fm" not in k
                      and "graph_only" not in k
                      and "cov_only" not in k}
        if candidates:
            best_key = min(candidates, key=lambda k: candidates[k]["metrics"]["MAE_masked"])
            entry = candidates[best_key]
            mae = _get_mae(entry)
            tr, inf = _get_times(entry)
            order_tag = best_key.split(f"_order_{mode}_timesfm25_")[1].replace("_all_cov", "").replace("_", " ")
            if mae is not None:
                rows.append({"label": "TimesFM + best", "mae": mae,
                             "train_time": tr, "infer_time": inf,
                             "group": "fm", "note": order_tag})

        if not rows:
            continue

        df = pd.DataFrame(rows)

        for time_col, time_label, fname_suffix in [
            ("train_time", "Training time (s)", "train"),
            ("infer_time", "Inference time (s)", "infer"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))

            for _, row in df.iterrows():
                lbl = row["label"]
                st  = STYLE.get(lbl, {"color": "grey", "marker": "o", "zorder": 3})
                fst = FM_LINESTYLE.get(lbl)

                if fst:
                    ax.scatter(row[time_col], row["mae"],
                               marker=st["marker"], zorder=st["zorder"],
                               edgecolors=fst["edgecolors"], facecolors=fst["facecolors"],
                               s=fst["s"], linewidths=1.5, label=lbl)
                else:
                    ax.scatter(row[time_col], row["mae"],
                               marker=st["marker"], color=st["color"],
                               zorder=st["zorder"], s=80, label=lbl)

            # deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h
            ax.legend(seen.values(), seen.keys(),
                      fontsize=8, loc="upper right", framealpha=0.85)

            ax.set_xlabel(time_label, fontsize=10)
            ax.set_ylabel("MAE (masked)", fontsize=10)
            ax.set_title(f"{MODE_LABELS[mode]} — MAE vs {time_label.split(' (')[0].lower()}",
                         fontsize=11)
            ax.grid(True, alpha=0.4)
            fig.tight_layout()

            out_path = f"{save_dir}/benchmark_timing_{fname_suffix}_{mode}.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"✓ Saved: {out_path}")


def get_results(results_file: str = "results_server/results.json",
        save_dir: str = "results_server/plots", filter_date = None):
    """
    Main function to generate all plots
    
    Args:
        results_file: Path to JSON results file
        save_dir: Directory to save plots
    """
    print(f"\nLoading results from: {results_file}")
    
    # Load and extract results
    results = load_results(results_file)
    print(f"Total models in results: {len(results)}")
    
    # filter by date if specified
    if filter_date:
        results_filtered = {}
        for model_name, res in results.items():
            creation_time = res.get("creation_time", "2026-01-01")
            if pd.to_datetime(creation_time) >= pd.to_datetime(filter_date):
                results_filtered[model_name] = res
        print(f"Models after date filter ({filter_date}): {len(results_filtered)}")
        results = results_filtered
    
    df = extract_module_order_results(results)
    
    if df.empty:
        print("\n❌ No module order experiments found!")
        print("\nTroubleshooting:")
        print("1. Make sure you've run module order ablation experiments")
        print("2. Model names should contain 'order_' (e.g., 'SpatialFM_bus_order_past_graph_future')")
        print("3. Or covariate_config should be one of: past_graph_future, graph_past_future, etc.")
        return
    
    print(f"\nFound {len(df)} module order experiments:")
    print(f"  Modes: {df['mode'].unique()}")
    #print(f"  Orderings: {df['order'].unique()}")
    
    # Generate all plots
    print(f"\nGenerating plots in: {save_dir}")
    print("-" * 80)
    
    plot_mae_comparison(df, save_dir)
    plot_training_curves(df, save_dir)
    plot_heatmap_mode_order(df, save_dir)
    plot_improvement_percentage(df, save_dir)
    plot_time_vs_performance(df, save_dir)
    summary_df = generate_summary_table(df, save_dir)
    
    print("-" * 80)
    print(f"\n✓ All plots saved to: {save_dir}/")
    
    # Print text summary (per mode and per FM)
    print_text_summary(df)
    generate_fm_text_summaries(results_file, save_dir, results=results, filter_date=filter_date)
    generate_mode_wide_tables(results_file, save_dir, results=results, filter_date=filter_date)
    generate_mode_wide_tables(results_file, save_dir, results=results, filter_date=filter_date, show_std=True)
    generate_benchmark_table(results_file, save_dir, results=results, filter_date=filter_date)
    generate_benchmark_timing_plots(results_file, save_dir, results=results, filter_date=filter_date)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")