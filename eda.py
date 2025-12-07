import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data File Paths ---
# NOTE: Update paths based on the local environment
DATA_PATH_BASE = r'C:\Users\schaf\Documents\International Exchange ICAM 5\Cranfield courses\Applied-Artificial-Intelligence-2025-26--2025-Oct-13_08-31-41-327\Statistical Learning Methods\Assignment data and codes\data'
TRAIN_FILE = 'train_selected.csv'

# --- Feature Selection ---
SENSOR_COLS = ['s1', 's2', 's3', 's4']

# Capping RUL prevents the model from learning from "infinite" health states
RUL_CAP = 125 

# Cycle index assumed to mark the onset of significant degradation
FAILURE_THRESHOLD_CYCLE = 120 

# --- Visualization Settings ---
COLOR_RAW = 'darkblue'    # Color for raw data density
COLOR_HEALTHY = 'skyblue' # Color for healthy phase
COLOR_FAULTY = 'coral'    # Color for degrading phase
ALPHA_DENSITY = 0.1       # Transparency for scatter plots (handling overplotting)



def load_and_prepare_data(filepath):
    """
    Loads the training dataset, cleans invalid rows, and ensures correct data types.
    
    Args:
        filepath (str): Full path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for analysis.
    """
    print(f"Loading data from: {filepath}...")
    
    # Define column schema
    column_names = ['engine_id', 'cycle'] + SENSOR_COLS + ['RUL', 'label_bnc']
    
    df = pd.read_csv(filepath, 
                     header=0, 
                     sep=';', 
                     decimal=',', 
                     names=column_names, 
                     engine='python') 
    
    initial_rows = len(df)
    
    # --- Cleaning Step: Remove header repetitions in body ---
    df = df[df['cycle'] != 'cycle']
    
    # --- Type Conversion ---
    # Coerce errors to NaN to identify non-numeric garbage data
    cols_to_numeric = ['engine_id', 'cycle', 'RUL'] + SENSOR_COLS
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with NaN values resulting from conversion
    df.dropna(subset=cols_to_numeric, inplace=True)
    
    # Convert structural columns to Integers
    df['engine_id'] = df['engine_id'].astype('Int64')
    df['cycle'] = df['cycle'].astype('Int64')
    df['RUL'] = df['RUL'].astype('Int64')
    
    # Create a capped RUL for correlation analysis later
    df['RUL_Capped'] = df['RUL'].apply(lambda x: min(x, RUL_CAP))
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"INFO: Data Cleaning removed {rows_removed} invalid rows.")
        
    print(f"SUCCESS: Data loaded. Total Rows: {len(df)}, Unique Engines: {df['engine_id'].nunique()}")
    return df

def plot_correlation_matrix(df):
    """
    Plots a heatmap showing Pearson correlation coefficients between 
    Cycles, Sensors, and RUL targets.
    """
    cols_to_correlate = ['cycle'] + SENSOR_COLS + ['RUL_Capped', 'RUL'] 
    
    # Ensure numeric types for correlation calculation
    corr_matrix = df[cols_to_correlate].apply(pd.to_numeric, errors='coerce').corr()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        cbar=True, 
        linewidths=.5, 
        linecolor='black'
    )
    plt.tight_layout()
    plt.show()


def plot_target_variable_distribution(df):
    """
    Plots the relationship between Sensor Values (X-axis) and Uncapped RUL (Y-axis).
    
    Design Choice:
    - Uses a shared Y-axis label to reduce clutter.
    - No main title (handled by report caption).
    - Large 'hspace' to prevent text overlap.
    """
    fig, axes = plt.subplots(nrows=len(SENSOR_COLS), ncols=1, figsize=(12, 18), sharex=False)
    
    for i, sensor in enumerate(SENSOR_COLS):
        ax = axes[i]
        
        # Scatter plot of Uncapped RUL
        ax.scatter(df[sensor], df['RUL'], color='darkred', alpha=ALPHA_DENSITY, s=10)
        
        ax.set_title(f'Sensor {sensor} vs. RUL (Target)', fontsize=14)
        ax.set_xlabel(f'Sensor {sensor} Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
            
    # Add a single centered Y-axis label for the entire figure
    fig.text(0.02, 0.5, 'Uncapped Cycle RUL', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    # Layout adjustment to accommodate the manual label
    plt.tight_layout(rect=[0.05, 0.01, 1, 0.98])
    plt.subplots_adjust(hspace=0.6) 
    plt.show()


def plot_raw_sensor_behavior(df):
    """
    Plots Raw Sensor Time-Series for ALL engines.
    
    Purpose: 
    To visualize the natural degradation trend and noise levels 
    before applying any thresholding logic.
    """
    mean_trend = df.groupby('cycle')[SENSOR_COLS].mean()
    
    fig, axes = plt.subplots(nrows=len(SENSOR_COLS), ncols=1, figsize=(14, 18), sharex=True)
    
    for i, sensor in enumerate(SENSOR_COLS):
        ax = axes[i]
        
        # 1. Plot individual engine cycles (Raw Data)
        ax.scatter(df['cycle'], df[sensor], 
                   color=COLOR_RAW,      # Dark Blue for contrast
                   alpha=ALPHA_DENSITY,  # Low alpha to visualize density
                   s=5,      
                   label='Individual Engine Cycles')
               
        # 2. Plot Mean Trend Line
        ax.plot(mean_trend.index, mean_trend[sensor], 
                color='black', linewidth=3, label='Mean Trend')
        
        # Styling
        ax.set_title(f'Sensor {sensor} over Engine Cycles', fontsize=12)
        ax.set_ylabel('Sensor Value (Units)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Legend: Remove duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {l: h for h, l in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=10)
        
        # Dynamic Axis Scaling
        # Focus Y-axis on the 1st and 99th percentile to hide extreme outliers
        q01 = df[sensor].quantile(0.01)
        q99 = df[sensor].quantile(0.99)
        padding = (q99 - q01) * 0.1 
        if pd.notna(q01) and pd.notna(q99):
            ax.set_ylim(q01 - padding, q99 + padding)
            
    # Set X-axis ticks
    max_cycle = int(df['cycle'].max())
    axes[-1].set_xticks(np.arange(0, max_cycle + 25, 25))
    axes[-1].set_xlim(0, max_cycle + 5)
    axes[-1].set_xlabel('Time in Engine Cycles')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def plot_sensor_degradation_analysis(df):
    """
    Plots Sensor Time-Series split by the Failure Threshold.
    
    Features:
    - Color coding (Healthy vs Degrading).
    - Variance Cone (Mean +/- 2 Std Dev) to show increasing noise (heteroscedasticity).
    - Threshold vertical line.
    """
    mean_trend = df.groupby('cycle')[SENSOR_COLS].mean()
    std_trend = df.groupby('cycle')[SENSOR_COLS].std()
    
    fig, axes = plt.subplots(nrows=len(SENSOR_COLS), ncols=1, figsize=(14, 18), sharex=True)
    
    # Split data based on hypothesis
    early_life = df[df['cycle'] < FAILURE_THRESHOLD_CYCLE]
    late_life = df[df['cycle'] >= FAILURE_THRESHOLD_CYCLE]
    
    for i, sensor in enumerate(SENSOR_COLS):
        ax = axes[i]
        
        # 1. Plot Healthy Phase
        ax.scatter(early_life['cycle'], early_life[sensor], 
                   color=COLOR_HEALTHY, alpha=ALPHA_DENSITY, s=5,      
                   label=f'Healthy (< {FAILURE_THRESHOLD_CYCLE})')
        
        # 2. Plot Degrading Phase
        ax.scatter(late_life['cycle'], late_life[sensor], 
                   color=COLOR_FAULTY, alpha=ALPHA_DENSITY, s=5,      
                   label=f'Degrading (>= {FAILURE_THRESHOLD_CYCLE})')
               
        # 3. Plot Mean Trend
        ax.plot(mean_trend.index, mean_trend[sensor], 
                color='black', linewidth=2, label='Mean Trend')
        
        # 4. Plot Variance Cone (Mean +/- 2*Std)
        # Visualizes the widening of noise distribution as engine degrades
        mean_val = mean_trend[sensor]
        std_val = std_trend[sensor].fillna(0) 
        ax.fill_between(mean_trend.index, 
                        mean_val - 2 * std_val, 
                        mean_val + 2 * std_val, 
                        color='black', alpha=0.15, 
                        label='Spread (Mean ± 2$\sigma$)')
        
        # 5. Threshold Line
        ax.axvline(x=FAILURE_THRESHOLD_CYCLE, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({FAILURE_THRESHOLD_CYCLE})')
        
        # Styling
        ax.set_title(f'Sensor {sensor} over Engine Cycles', fontsize=12)
        ax.set_ylabel('Sensor Value (Units)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {l: h for h, l in zip(handles, labels)}
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=10)
        
        # Axis Scaling (Same logic as raw plot)
        q01 = df[sensor].quantile(0.01)
        q99 = df[sensor].quantile(0.99)
        padding = (q99 - q01) * 0.1 
        if pd.notna(q01) and pd.notna(q99):
            ax.set_ylim(q01 - padding, q99 + padding)
            
    # Set X-axis ticks
    max_cycle = int(df['cycle'].max())
    axes[-1].set_xticks(np.arange(0, max_cycle + 25, 25))
    axes[-1].set_xlim(0, max_cycle + 5)
    axes[-1].set_xlabel('Time in Engine Cycles')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


# MAIN EXECUTION FLOW

if __name__ == '__main__':
    # Construct full file path
    full_train_filepath = f"{DATA_PATH_BASE}\\{TRAIN_FILE}"
    
    # 1. Load Data
    df_train_eda = load_and_prepare_data(full_train_filepath)
    
    if not df_train_eda.empty:
        print("\n--- 1. Generating Raw Sensor Behavior Plot ---")
        plot_raw_sensor_behavior(df_train_eda)
        
        print("\n--- 2. Generating Degradation Analysis Plot (Split & Variance) ---")
        plot_sensor_degradation_analysis(df_train_eda) 
        
        print("\n--- 3. Generating RUL Target Distribution Plot ---")
        plot_target_variable_distribution(df_train_eda)
        
        print("\n--- 4. Generating Correlation Matrix ---")
        plot_correlation_matrix(df_train_eda)
        
        print("\nAnalysis Complete.")
    else:

        print("\nERROR: Data frame is empty. Please check the data path and file format.")

