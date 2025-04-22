import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np # For checking class distribution

# --- Configuration ---
DATASET_PATH = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\cleaned\twcs_sample_cleaned.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\data-train-test"
TARGET_COLUMN = 'inbound' # The column we need to stratify by

TEST_SPLIT_RATIO = 0.20 # 20% for the test set
RANDOM_STATE_SEED = 42 # For reproducible splits - a common choice

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory ensured: {OUTPUT_DIR}")

# --- Load Data ---
print(f"Loading dataset from: {DATASET_PATH}")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Data Pre-check and Target Preparation ---
print("Performing pre-split checks...")
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    exit()

# Ensure target column is suitable for stratification (numeric/boolean)
# Re-apply conversion logic just in case this script is run independently
if df[TARGET_COLUMN].dtype == 'bool':
    print(f"Converting boolean target column '{TARGET_COLUMN}' to integer (0/1).")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
elif df[TARGET_COLUMN].dtype == 'object':
    print(f"Attempting to convert object target column '{TARGET_COLUMN}' to integer (assuming TRUE/FALSE strings).")
    original_nan_count = df[TARGET_COLUMN].isnull().sum()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.upper().map({'TRUE': 1, 'FALSE': 0})
    final_nan_count = df[TARGET_COLUMN].isnull().sum()

    if final_nan_count > original_nan_count:
        print(f"Warning: {final_nan_count - original_nan_count} values in '{TARGET_COLUMN}' could not be converted to 0/1.")
        # Decide how to handle NaNs: drop or impute. Dropping is safer if few.
        rows_before_drop = df.shape[0]
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        print(f"Dropped {rows_before_drop - df.shape[0]} rows with unparseable target values. New shape: {df.shape}")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int) # Ensure type is int after potential map/dropna
    elif df[TARGET_COLUMN].isnull().any():
         print(f"Warning: Column '{TARGET_COLUMN}' still contains NaN values after conversion attempt. Check data.")
         # Handle remaining NaNs if necessary (e.g., dropna)
         df.dropna(subset=[TARGET_COLUMN], inplace=True)
         df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

# Check if target is now numeric
if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
     print(f"Error: Target column '{TARGET_COLUMN}' is not numeric after preparation, cannot stratify reliably.")
     exit()

print(f"Target column '{TARGET_COLUMN}' prepared for stratification.")

# Display initial class distribution
print("\nOriginal dataset class distribution:")
print(df[TARGET_COLUMN].value_counts(normalize=True))

# --- Perform Stratified Train-Test Split ---
print(f"\nSplitting data: {100*(1-TEST_SPLIT_RATIO):.0f}% train / {100*TEST_SPLIT_RATIO:.0f}% test...")
print(f"Using stratification based on '{TARGET_COLUMN}' column.")
print(f"Using random_state = {RANDOM_STATE_SEED} for reproducibility.")

try:
    train_df, test_df = train_test_split(
        df, # The entire dataframe to split
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE_SEED,
        stratify=df[TARGET_COLUMN], # Key parameter for balanced classes
        shuffle=True # Default, but good to be explicit
    )
except ValueError as ve:
    if "The least populated class" in str(ve):
        print(f"\nError during stratification: {ve}")
        print("This usually means one class has too few samples to be split according to the test_size.")
        print("Consider using a smaller test_size, getting more data, or techniques like upsampling/downsampling (applied *after* splitting).")
    else:
        print(f"\nAn unexpected error occurred during splitting: {ve}")
    exit()
except Exception as e:
    print(f"\nAn unexpected error occurred during splitting: {e}")
    exit()


# --- Verification ---
print("\nSplit verification:")
print(f"  Training set shape: {train_df.shape}")
print(f"  Test set shape:     {test_df.shape}")

print("\nTraining set class distribution:")
print(train_df[TARGET_COLUMN].value_counts(normalize=True))

print("\nTest set class distribution:")
print(test_df[TARGET_COLUMN].value_counts(normalize=True))

# Check if distributions are similar (they should be due to stratification)
train_dist = train_df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
test_dist = test_df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
if np.allclose(train_dist, test_dist, atol=0.01): # Allow small tolerance
    print("\nClass distributions appear well-stratified between train and test sets.")
else:
    print("\nWarning: Class distributions differ significantly between train and test sets. Check stratification.")

# --- Save Split Data ---
train_output_path = os.path.join(OUTPUT_DIR, 'train_data.csv')
test_output_path = os.path.join(OUTPUT_DIR, 'test_data.csv')

print(f"\nSaving training data to: {train_output_path}")
train_df.to_csv(train_output_path, index=False, encoding='utf-8')

print(f"Saving test data to: {test_output_path}")
test_df.to_csv(test_output_path, index=False, encoding='utf-8')

print("-" * 30)
print("Data splitting complete!")
print(f"Train and test CSV files saved in: {OUTPUT_DIR}")
print("-" * 30)