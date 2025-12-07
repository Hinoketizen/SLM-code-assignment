import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 

print("Libraries imported successfully.")

# --- 1. CONFIGURATION ---
FEATURES = ['cycle', 's1', 's2', 's3', 's4']
RUL_CAP = 125 

# Update this path if necessary
DATA_PATH_BASE = r'C:\Users\schaf\Documents\International Exchange ICAM 5\Cranfield courses\Applied-Artificial-Intelligence-2025-26--2025-Oct-13_08-31-41-327\Statistical Learning Methods\Assignment data and codes\data'
TRAIN_FILE = 'train_selected.csv'
TEST_FILE = 'test_selected.csv'
TRUTH_FILE = 'PM_truth.txt'

# --- 2. BASE LOADING FUNCTION ---
def load_data(filepath):
    """
    Loads data handling the ';' separator and ',' decimal.
    """
    # Read column names from the first row
    header_names = pd.read_csv(filepath, sep=';', decimal=',', nrows=0).columns.tolist()
    
    df = pd.read_csv(filepath,
                     header=0, 
                     sep=';',
                     decimal=',',
                     names=header_names, 
                     engine='python')
    
    # Data cleaning: ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(inplace=True)
    return df

# --- 3. CUSTOM REPORT FUNCTION ---
def print_custom_report(y_true, y_pred, model_name):
    """
    Prints a custom classification report:
    - Removes 'accuracy'
    - Removes 'weighted avg'
    - Renames 'support' to 'units'
    """
    print(f"\n--- Custom Classification Report for {model_name} ---")
    
    # Generate report as a dictionary
    report_dict = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
    
    # Convert to DataFrame for easy manipulation
    df_report = pd.DataFrame(report_dict).transpose()
    
    # 1. Drop accuracy and weighted avg
    cols_to_drop = ['accuracy', 'weighted avg']
    for col in cols_to_drop:
        if col in df_report.index:
            df_report.drop(col, inplace=True)
            
    # 2. Rename 'support' to 'units'
    df_report.rename(columns={'support': 'units'}, inplace=True)
    
    # Convert 'units' column to integer for display
    df_report['units'] = df_report['units'].astype(int)
    
    # Clean display
    print(df_report)


# --- 4. TRAIN AND EVALUATE FUNCTION ---
def train_and_evaluate_classifier(model, model_name, X_train_scaled, y_train, X_test_scaled, y_test, use_grid_search=False):
    """
    Trains a classifier (or GridSearch) and displays metrics.
    """
    print(f"\n--- Model Training: {model_name} ---")
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    # If GridSearch is used, retrieve the best model for prediction
    if use_grid_search:
        print(f"Best parameters found: {model.best_params_}")
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test_scaled)
        
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # --- Custom Evaluation ---
    print_custom_report(y_test, y_pred, model_name)

    # --- Confusion Matrix Display ---
    print(f"Generating Confusion Matrix for {model_name}...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.show()

# --- MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    
    # --- 1. Load and Prepare TRAINING Data ---
    print("Loading training data (train_selected.csv)...")
    full_train_filepath = f"{DATA_PATH_BASE}\\{TRAIN_FILE}"
    df_train = load_data(full_train_filepath)

    X_train = df_train[FEATURES]
    y_train = df_train['label_bnc'].astype('int')
    
    print(f"Training data: {X_train.shape[0]} rows.")
    print("\nClass distribution (Train Set):")
    print(y_train.value_counts(normalize=True))

    # --- 2. Load and Prepare TEST Data ---
    print("\nLoading test data (test_selected.csv)...")
    full_test_filepath = f"{DATA_PATH_BASE}\\{TEST_FILE}"
    df_test = load_data(full_test_filepath)
    X_test = df_test[FEATURES]

    print("\nLoading ground truth (PM_truth.txt)...")
    full_truth_filepath = f"{DATA_PATH_BASE}\\{TRUTH_FILE}"
    df_truth = pd.read_csv(full_truth_filepath, header=None, names=['RUL_true'])
    
    # Create test labels: 1 if RUL <= 30 (imminent failure), else 0
    y_test = df_truth['RUL_true'].apply(lambda rul: 1 if rul <= 30 else 0).astype('int')
    
    print(f"Test data: {X_test.shape[0]} rows.")
    print("\nClass distribution (Test Set):")
    print(y_test.value_counts(normalize=True))

    # --- 3. Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures scaled.")

    # --- 4. Model Execution: SVC with GridSearchCV ---
    print("\n" + "="*50)
    print(" MODEL: SVC WITH GRID SEARCH (OPTIMIZATION)")
    print("="*50)
    
    # Define parameter grid to test
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf'] 
    }
    
    base_svc = SVC(class_weight='balanced', random_state=42)
    
    # GridSearch Configuration
    grid_search = GridSearchCV(estimator=base_svc, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    
    # Launch optimization and evaluation
    train_and_evaluate_classifier(grid_search, "SVMC (RBF kernel) Optimised (GridSearch)", X_train_scaled, y_train, X_test_scaled, y_test, use_grid_search=True)

    print("\nClassification analysis completed.")