import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
import time
import os # Added os import, it was used later but not imported

# Import necessary types for hinting
from typing import Tuple, List, Dict, Any # <-- Added this import

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --- Configuration ---


INPUT_DIR = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\Inbound vs. Outbound Tweet Classifier\data\feature engineering\log_reg_feature_engr_testing")


# Output Directory for saving the trained model and results
# *** UPDATED OUTPUT DIRECTORY ***
OUTPUT_DIR = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\Inbound vs. Outbound Tweet Classifier\data\model_results\testing" )
# Alternatively, using raw string:
# OUTPUT_DIR = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\model_results")


# File Names (ensure these match the output of the feature engineering script)
FEATURES_FILE = "engineered_features_train.parquet"
FEATURE_NAMES_FILE = "feature_names.json"
SCALER_FILE = "numerical_scaler.joblib" 
VECTORIZER_FILE = "tfidf_vectorizer.joblib" 

# Model & Evaluation Parameters
TARGET_COLUMN = 'inbound'  # Name of the target variable column
TEST_SPLIT_SIZE = 0.20    # Proportion of data to use for validation set
RANDOM_STATE = 42         # For reproducibility of train/test split and model training

# Logistic Regression Hyperparameters (Baseline settings)
LOGREG_PARAMS = {
    'C': 1.0,                   # Regularization strength (default)
    'solver': 'saga',          # Good solver for potentially large datasets, handles L1/L2
    'penalty': 'l2',           # Standard L2 regularization
    'class_weight': 'balanced', # Crucial for potentially imbalanced datasets
    'max_iter': 1000,           # Increase iterations for convergence assurance
    'random_state': RANDOM_STATE,
    'n_jobs': -1                # Use all available CPU cores
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_artifacts(input_dir: Path) -> Tuple[pd.DataFrame, List[str], Any, Any]:
    """Loads the engineered features and preprocessing objects."""
    logging.info(f"Loading artifacts from: {input_dir}")

    features_path = input_dir / FEATURES_FILE
    feature_names_path = input_dir / FEATURE_NAMES_FILE
    scaler_path = input_dir / SCALER_FILE
    vectorizer_path = input_dir / VECTORIZER_FILE

    # Check if files exist
    if not all([features_path.exists(), feature_names_path.exists(),
                scaler_path.exists(), vectorizer_path.exists()]):
        logging.error("One or more required artifact files are missing.")
        raise FileNotFoundError("Missing input artifact files.")

    try:
        df_features = pd.read_parquet(features_path)
        logging.info(f"Loaded features data. Shape: {df_features.shape}")

        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded {len(feature_names)} feature names.")

        # Load scaler and vectorizer - mainly to demonstrate pipeline completeness
        # They aren't used for transformation here as features are pre-computed
        scaler = joblib.load(scaler_path)
        logging.info("Loaded numerical scaler object.")
        vectorizer = joblib.load(vectorizer_path)
        logging.info("Loaded TF-IDF vectorizer object.")

        return df_features, feature_names, scaler, vectorizer

    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")
        raise

def train_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         model_params: Dict[str, Any],
                         results_output_dir: Path) -> Tuple[LogisticRegression, Dict[str, float]]: # Added results_output_dir
    """Trains Logistic Regression and evaluates on the validation set."""
    logging.info("Training Logistic Regression model...")
    logging.info(f"Model parameters: {model_params}")

    model = LogisticRegression(**model_params)

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    logging.info(f"Model training completed in {end_time - start_time:.2f} seconds.")

    logging.info("Evaluating model on validation set...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] # Probability of positive class (inbound=1)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    # Use pos_label=1 assuming 'inbound' is the positive class (often the case for detection tasks)
    # Adjust if 'outbound' (0) is your positive class of interest
    precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_pred_proba) # Requires probabilities of the positive class
    conf_matrix = confusion_matrix(y_val, y_pred)

    logging.info("\n--- Validation Set Evaluation Metrics (Positive Class: Inbound=1) ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC AUC:   {roc_auc:.4f}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")

    # Detailed classification report
    report = classification_report(y_val, y_pred, target_names=['Outbound (0)', 'Inbound (1)'], zero_division=0)
    logging.info(f"Classification Report:\n{report}")

    metrics = {
        'accuracy': accuracy,
        'precision_inbound_1': precision, # Make metric names specific
        'recall_inbound_1': recall,
        'f1_score_inbound_1': f1,
        'roc_auc': roc_auc,
        'training_time_seconds': end_time - start_time
        # You could add metrics for class 0 here if needed
    }

    # --- Save classification report and metrics to file ---
    report_path = results_output_dir / "classification_report.txt"
    metrics_path = results_output_dir / "validation_metrics.json"
    os.makedirs(results_output_dir, exist_ok=True) # Ensure dir exists before writing

    # Save text report
    try:
        with open(report_path, 'w') as f:
            f.write("Validation Set Evaluation Metrics (Positive Class: Inbound=1)\n")
            f.write("===========================================================\n")
            # Write metrics dictionary first for easy parsing
            for key, value in metrics.items():
                 if isinstance(value, float):
                     f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                 else: # Handle non-float values like training time
                     f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write(f"\nConfusion Matrix:\n{conf_matrix}\n")
            f.write(f"\nClassification Report:\n{report}\n")
        logging.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logging.error(f"Error saving classification report: {e}")

    # Save metrics as JSON
    try:
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native python types for JSON serialization
            serializable_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=4)
        logging.info(f"Validation metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics JSON: {e}")

    return model, metrics

def save_model(model: LogisticRegression, output_dir: Path, filename: str = "logreg_baseline_model.joblib"):
    """Saves the trained model to disk."""
    logging.info(f"Saving trained model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    model_path = output_dir / filename
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model successfully saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

# --- Main Execution ---

def main():
    """Orchestrates the model training and evaluation process."""
    logging.info("--- Starting Baseline Model Training Script ---")

    try:
        # 1. Load Artifacts
        df, feature_names, _, _ = load_artifacts(INPUT_DIR) # Scaler/Vectorizer not needed for training logic here

        # 2. Prepare Data
        if TARGET_COLUMN not in df.columns:
            logging.error(f"Target column '{TARGET_COLUMN}' not found in the dataframe.")
            return
        if not all(f_name in df.columns for f_name in feature_names):
            logging.warning("Mismatch between feature names list and columns in the dataframe. Using intersection.")
            # Find missing/extra columns for debugging
            missing_in_df = [f for f in feature_names if f not in df.columns]
            extra_in_df = [c for c in df.columns if c not in feature_names and c != TARGET_COLUMN and not c.startswith('feat_') and not c.startswith('tfidf_')] # Refine check
            logging.warning(f"Features in list but not in DataFrame: {missing_in_df}")
            logging.warning(f"Non-feature columns in DataFrame (ignoring target): {extra_in_df}") # Usually just ID

            # Proceed with only the intersection of features
            valid_features = [f for f in feature_names if f in df.columns]
            if not valid_features:
                 logging.error("No valid features found to proceed.")
                 return
            logging.info(f"Proceeding with {len(valid_features)} features found in both list and dataframe.")
            feature_names = valid_features # Update feature_names to only include valid ones


        X = df[feature_names]
        y = df[TARGET_COLUMN]

        # Check for NaNs/Infs in features, which LogReg can't handle
        if X.isnull().any().any(): # More robust check for NaNs
             nan_cols = X.columns[X.isnull().any()].tolist()
             logging.warning(f"NaN values found in feature matrix X (Columns: {nan_cols}). Filling with 0.")
             X = X.fillna(0) # Simple imputation strategy for baseline
        if np.any(np.isinf(X.values)): # Check underlying numpy array for infinity
             logging.warning("Infinite values found in feature matrix X. Replacing with large finite numbers.")
              # Replace inf with a large number, -inf with a small number
             X = X.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])


        logging.info(f"Features matrix shape after prep: {X.shape}")
        logging.info(f"Target vector shape: {y.shape}")
        logging.info(f"Class distribution:\n{y.value_counts(normalize=True)}")


        # 3. Split Data into Training and Validation Sets
        logging.info(f"Splitting data into training and validation sets (Test Size: {TEST_SPLIT_SIZE})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=y  # Important for potentially imbalanced classes
        )
        logging.info(f"Training set shape:   X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
        logging.info(f"Validation set class distribution:\n{y_val.value_counts(normalize=True)}")


        # 4. Train and Evaluate Model
        # Pass the specific OUTPUT_DIR for saving reports/metrics
        model, metrics = train_evaluate_model(X_train, y_train, X_val, y_val, LOGREG_PARAMS, results_output_dir=OUTPUT_DIR)

        # 5. Save the Trained Model (also uses OUTPUT_DIR)
        save_model(model, OUTPUT_DIR)

    except FileNotFoundError:
        logging.error("Script aborted due to missing input files.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log full traceback

    logging.info("--- Baseline Model Training Script Finished ---")


if __name__ == "__main__":
    main() 