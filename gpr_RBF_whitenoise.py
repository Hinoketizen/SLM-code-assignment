import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Gaussian Process imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

print("Libraries imported successfully.")

# --- 1. CONFIGURATION --- 
# (Copied from eda_rul_analysis.py for notebook portability)
SENSOR_COLS = ['s1', 's2', 's3', 's4']
# --- NEW: Define the full feature set including 'cycle' ---
FEATURE_COLS = ['cycle'] + SENSOR_COLS 
RUL_CAP = 125 
DATA_PATH_BASE = r'C:\Users\schaf\Documents\International Exchange ICAM 5\Cranfield courses\Applied-Artificial-Intelligence-2025-26--2025-Oct-13_08-31-41-327\Statistical Learning Methods\Assignment data and codes\data'
TRAIN_FILE = 'train_selected.csv'
TEST_FILE = 'test_selected.csv'
TRUTH_FILE = 'PM_truth.txt'

# --- 2. DATA LOADING AND RUL CALCULATION ---

def load_and_prepare_data(filepath):
    """
    Loads training data and uses the provided 'ttf' column as the RUL.
    RUL capping is calculated for later modeling.
    """
    
    # Define the expected columns, including the provided ttf and label
    COLUMN_NAMES = ['engine_id', 'cycle'] + SENSOR_COLS + ['RUL', 'label_bnc']
    
    df = pd.read_csv(filepath, 
                     header=0, 
                     sep=';',
                     decimal=',',
                     names=COLUMN_NAMES, 
                     engine='python') 
    
    # --- ROBUST DATA CLEANING AND TYPE CONVERSION ---
    initial_rows = len(df)
    
    # Drop the extra header row that might be read
    df = df[df['cycle'] != 'cycle']
    
    # 1. Ensure core ID/target columns are numeric
    df['engine_id'] = pd.to_numeric(df['engine_id'], errors='coerce')
    df['cycle'] = pd.to_numeric(df['cycle'], errors='coerce')
    df['RUL'] = pd.to_numeric(df['RUL'], errors='coerce') # Use the provided RUL/TTF
    
    # 2. Convert sensor columns to numeric
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 3. Drop any rows where key data is missing/invalid
    df.dropna(subset=['engine_id', 'cycle', 'RUL'] + SENSOR_COLS, inplace=True)
    
    # Final conversion to integer types
    df['engine_id'] = df['engine_id'].astype('Int64')
    df['cycle'] = df['cycle'].astype('Int64')
    df['RUL'] = df['RUL'].astype('Int64')
    
    final_rows = len(df)
    if initial_rows != final_rows:
        print(f"INFO: Removed {initial_rows - final_rows} rows during cleaning.")
    
    # --- RUL CAPPING ---
    # Calculate RUL Capped for later modeling/comparison
    df['RUL_Capped'] = df['RUL'].apply(lambda x: min(x, RUL_CAP))
    
    return df

# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    
    # --- 3. Feature, Target, and Test Set Preparation ---
    print("Loading and preparing data...")
    # 1. Load Training Data
    full_train_filepath = f"{DATA_PATH_BASE}\\{TRAIN_FILE}"
    df_train = load_and_prepare_data(full_train_filepath)

    # 2. Define X and y
    # --- EDIT: Use all 5 features (cycle + 4 sensors) ---
    X = df_train[FEATURE_COLS] 
    y = df_train['RUL_Capped'] # <-- Target is RUL_Capped

    # 3. Sub-sample the data for GPR training
    SAMPLE_SIZE = 2500
    np.random.seed(42) # for reproducibility
    sample_indices = np.random.choice(X.index, SAMPLE_SIZE, replace=False)

    X_train_subset = X.loc[sample_indices]
    y_train_subset = y.loc[sample_indices]

    print(f"Training on a random subset of {SAMPLE_SIZE} data points.")

    # 4. Scale Training Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)

    # 5. Load and Prepare Test Data
    full_test_filepath = f"{DATA_PATH_BASE}\\{TEST_FILE}"
    df_test = pd.read_csv(full_test_filepath, sep=';', decimal=',', header=0)
    # Ensure correct column names even if test file has 'id'
    df_test.rename(columns={'id': 'engine_id'}, inplace=True)
    # --- EDIT: Use all 5 features (cycle + 4 sensors) ---
    X_test = df_test[FEATURE_COLS]

    # Load ground truth RUL values
    full_truth_filepath = f"{DATA_PATH_BASE}\\{TRUTH_FILE}"
    y_truth_df = pd.read_csv(full_truth_filepath, header=None, names=['RUL_True'])
    y_test = y_truth_df['RUL_True']

    # Apply the same RUL Capping to the test target
    y_test_capped = y_test.apply(lambda x: min(x, RUL_CAP))

    # Scale the test features using the *same scaler* fit on the training subset
    X_test_scaled = scaler.transform(X_test)

    print("Data preparation complete.")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"y_train_subset shape: {y_train_subset.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_test_capped shape: {y_test_capped.shape}")

    # --- 4. GPR Kernel Definition ---
    # Define the kernel for the GPR
    # 1. ConstantKernel * RBF: Models the non-linear degradation trend
    # 2. WhiteKernel: Models the noise (variability)
    # The RBF kernel will now have 5 length_scale parameters (one for each feature)
    n_features = len(FEATURE_COLS)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
    )

    print("\nGPR kernel defined:")
    print(kernel)

    # --- 5. Model Training ---
    print("\nInitializing GPR model...")
    gp_regressor = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=9, # Helps find a good hyperparameter optimum
        random_state=42
    )

    print("Training GPR model (this may take a few minutes)...")
    start_time = time.time()

    gp_regressor.fit(X_train_scaled, y_train_subset)

    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds.")

    # --- NEW: Print the final optimized kernel and its LML score ---
    print("\nFitted Kernel Parameters:")
    print(gp_regressor.kernel_) 
    print(f"\nFinal Log-Marginal-Likelihood: {gp_regressor.log_marginal_likelihood_value_:.3f}")
    # -------------------------------------------------------------

    # --- 6. Prediction and Evaluation ---
    print("\nMaking predictions on the test set...")
    start_time = time.time()

    # Get both the prediction (mean) and the uncertainty (std. deviation)
    y_pred, std_pred = gp_regressor.predict(X_test_scaled, return_std=True)

    end_time = time.time()
    print(f"Prediction complete in {end_time - start_time:.2f} seconds.")

    # Calculate metrics
    mae = mean_absolute_error(y_test_capped, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_capped, y_pred))
    r2 = r2_score(y_test_capped, y_pred)

    print("\n--- GPR Model Performance (on Test Set) ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} cycles")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} cycles")

    # --- 7. Visualization 1: Actual vs. Predicted (with Uncertainty) ---
    print("\nGenerating Visualization 1: Actual vs. Predicted...")
    plt.figure(figsize=(10, 8))

    # Calculate the 95% confidence interval
    confidence = 1.96 * std_pred

    plt.errorbar(y_test_capped, y_pred, 
                 yerr=confidence, 
                 fmt='o', 
                 ecolor='lightgray', 
                 elinewidth=2, 
                 capsize=0, 
                 alpha=0.6, 
                 label='95% Confidence Interval')

    plt.scatter(y_test_capped, y_pred, alpha=0.8, c='blue', label='Prediction')

    # Plot the ideal 45-degree line
    plt.plot([0, RUL_CAP], [0, RUL_CAP], 'r--', linewidth=2, label='Ideal Fit (y=x)')

    plt.title('GPR: Actual vs. Predicted RUL (with 95% Confidence)', fontsize=16)
    plt.xlabel('Actual RUL (Capped)', fontsize=12)
    plt.ylabel('Predicted RUL', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, RUL_CAP + 5)
    plt.ylim(0, RUL_CAP + 20) # Give space for error bars
    
    # --- 8. Visualization 2: Model Fit vs. Key Features (All 5 Features) ---
    print("Generating Visualization 2: Model Fit vs. All Sensor Features (Separate Plots)...")
    
    # --- EDIT: Create a 3x2 grid to hold all 5 feature plots ---
    # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 21))
    # fig.suptitle('GPR Model Fit vs. All Features', fontsize=20)
    
    # Flatten the axes array for easy iteration
    # axes_flat = axes.flatten()
    
    # Prepare the metrics text (it's the same for all plots)
    metrics_text = (
        f"Overall Model Performance:\n"
        f"R-squared (R2): {r2:.4f}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}"
    )

    # --- EDIT: Loop over all 5 FEATURE_COLS, creating a NEW figure for each ---
    for feature in FEATURE_COLS:
        # 1. Create a new, separate figure for this feature
        plt.figure(figsize=(10, 7)) # Good size for a single plot
        ax = plt.gca() # Get the current axis for this new figure

        # We need the unscaled sensor values from the test set for this plot
        feature_test = X_test[feature]

        # Sort the values for a clean plot
        sort_indices = np.argsort(feature_test)
        feature_test_sorted = feature_test.iloc[sort_indices]
        y_pred_sorted = y_pred[sort_indices]
        y_test_capped_sorted = y_test_capped.iloc[sort_indices]
        std_pred_sorted = std_pred[sort_indices]

        confidence_sorted = 1.96 * std_pred_sorted

        # Plot the actual data points
        ax.scatter(feature_test, y_test_capped, 
                    alpha=0.2, 
                    color='gray', 
                    label='Actual Test Data')

        # Plot the GPR's mean prediction
        ax.plot(feature_test_sorted, y_pred_sorted, 
                 color='blue', 
                 linewidth=2, 
                 label='GPR Mean Prediction')

        # Plot the confidence interval (the shaded region)
        ax.fill_between(feature_test_sorted, 
                         y_pred_sorted - confidence_sorted, 
                         y_pred_sorted + confidence_sorted,
                         color='skyblue', 
                         alpha=0.3, 
                         label='95% Confidence Interval')
        
        # Add the metrics box to the plot
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'GPR Model Fit vs. Feature: {feature}', fontsize=14)
        ax.set_xlabel(f'Feature: {feature} (Unscaled)', fontsize=12)
        ax.set_ylabel('RUL (Capped)', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add tight_layout to each individual figure
        plt.tight_layout()
    
    # --- NEW: Hide the last (empty) subplot ---
    # axes_flat[-1].axis('off') # No longer needed
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # No longer needed
    
    # Show both plots
    print("Displaying all 5 feature fit plots...")
    plt.show()

    print("\nScript execution complete.")