import os
import pandas as pd
import numpy as np
import string
import re
from typing import Tuple, List, Dict, Any

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Specifically good for social media

# Utilities
import joblib # For saving sklearn objects (like vectorizers, scalers)
import logging
from pathlib import Path


INPUT_CSV_PATH = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\Inbound vs. Outbound Tweet Classifier\data\processed\data-train-test\test_data.csv")


# Output Directory (Adjust if necessary)
# OUTPUT_DIR = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\feature engineering")
OUTPUT_DIR = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\Inbound vs. Outbound Tweet Classifier\data\feature engineering")


# Feature Engineering Parameters
TFIDF_MAX_FEATURES = 5000 # Max number of TF-IDF features (controls dimensionality)
TFIDF_NGRAM_RANGE = (1, 2) # Use unigrams and bigrams
TFIDF_MIN_DF = 3         # Ignore terms with document frequency lower than this
TFIDF_MAX_DF = 0.85      # Ignore terms with document frequency higher than this (potential corpus-specific stop words)

# Columns to use
TEXT_COLUMN = 'text'          # Original text for stats like length, punctuation, sentiment
CLEAN_TEXT_COLUMN = 'clean_text' # Preprocessed text for TF-IDF
TARGET_COLUMN = 'inbound'     # Boolean target variable
ID_COLUMN = 'tweet_id'        # To keep track of tweets

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def calculate_basic_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Calculates basic features from the original text column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the column containing the original tweet text.

    Returns:
        pd.DataFrame: DataFrame with new basic feature columns added.
                      Features: text_length, word_count, punctuation_count,
                                mention_count, hashtag_count, url_count.
    """
    logging.info(f"Calculating basic text features from column: '{text_col}'")

    # Ensure the text column is string type and handle potential NaNs
    df[text_col] = df[text_col].astype(str).fillna('')

    # 1. Text Length
    df['feat_text_length'] = df[text_col].apply(len)

    # 2. Word Count (simple split)
    df['feat_word_count'] = df[text_col].apply(lambda x: len(x.split()))

    # 3. Punctuation Count
    df['feat_punctuation_count'] = df[text_col].apply(
        lambda x: sum(1 for char in x if char in string.punctuation)
    )

    # 4. Mention Count (@)
    df['feat_mention_count'] = df[text_col].apply(lambda x: x.count('@'))

    # 5. Hashtag Count (#)
    df['feat_hashtag_count'] = df[text_col].apply(lambda x: x.count('#'))

    # 6. URL Count (simple check for 'http')
    df['feat_url_count'] = df[text_col].apply(lambda x: len(re.findall(r'http[s]?://', x)))

    logging.info("Basic text features calculated.")
    return df

def add_sentiment_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Adds VADER sentiment scores (compound, positive, negative, neutral)
    based on the original text column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the column containing the original tweet text.

    Returns:
        pd.DataFrame: DataFrame with new sentiment feature columns added.
    """
    logging.info(f"Calculating sentiment features using VADER from column: '{text_col}'")
    analyzer = SentimentIntensityAnalyzer()

    # Ensure the text column is string type and handle potential NaNs
    df[text_col] = df[text_col].astype(str).fillna('')

    sentiment_scores = df[text_col].apply(analyzer.polarity_scores)

    # Expand the dictionary of scores into separate columns
    sentiment_df = pd.json_normalize(sentiment_scores)
    sentiment_df.columns = [f'feat_sentiment_{col}' for col in sentiment_df.columns] # Add prefix

    # Join back with the original dataframe
    df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

    logging.info("Sentiment features calculated.")
    return df

def create_tfidf_features(df: pd.DataFrame, text_col: str, **tfidf_params) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Creates TF-IDF features from the specified text column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the column containing the text for TF-IDF (usually cleaned).
        **tfidf_params: Keyword arguments passed directly to TfidfVectorizer
                        (e.g., max_features, ngram_range, min_df, max_df).

    Returns:
        Tuple[pd.DataFrame, TfidfVectorizer]:
            - DataFrame containing the TF-IDF features.
            - The fitted TfidfVectorizer object.
    """
    logging.info(f"Creating TF-IDF features from column: '{text_col}'")
    logging.info(f"TF-IDF Parameters: {tfidf_params}")

    # Ensure the text column is string type and handle potential NaNs
    texts = df[text_col].astype(str).fillna('')

    vectorizer = TfidfVectorizer(stop_words='english', **tfidf_params)

    # Fit the vectorizer and transform the text data
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Convert the sparse matrix to a DataFrame for easier handling (can be memory intensive)
    # Consider using sparse matrices directly if memory becomes an issue
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=[f"tfidf_{feature}" for feature in vectorizer.get_feature_names_out()])

    logging.info(f"TF-IDF features created with shape: {tfidf_df.shape}")
    return tfidf_df, vectorizer

def scale_numerical_features(df: pd.DataFrame, features_to_scale: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Applies StandardScaler to specified numerical feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the features.
        features_to_scale (List[str]): List of column names to scale.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]:
            - DataFrame with specified columns scaled.
            - The fitted StandardScaler object.
    """
    logging.info(f"Scaling numerical features: {features_to_scale}")
    scaler = StandardScaler()
    df_scaled = df.copy() # Avoid modifying the original DataFrame slice

    # Fit and transform the specified columns
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    logging.info("Numerical features scaled.")
    return df_scaled, scaler


# --- Main Execution ---

def main():
    """
    Main function to orchestrate the feature engineering process.
    """
    logging.info("--- Starting Feature Engineering Script ---")

    # --- 1. Load Data ---
    logging.info(f"Loading data from: {INPUT_CSV_PATH}")
    try:
        df_train = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
        logging.info(f"Data loaded successfully. Shape: {df_train.shape}")
        logging.info(f"Columns: {df_train.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Ensure required columns exist
    required_cols = [ID_COLUMN, TEXT_COLUMN, CLEAN_TEXT_COLUMN, TARGET_COLUMN]
    if not all(col in df_train.columns for col in required_cols):
        logging.error(f"Missing required columns. Expected: {required_cols}, Found: {df_train.columns.tolist()}")
        return

    # Convert target 'inbound' boolean to integer (0 or 1) for modeling
    if df_train[TARGET_COLUMN].dtype == bool:
        df_train[TARGET_COLUMN] = df_train[TARGET_COLUMN].astype(int)
        logging.info(f"Converted target column '{TARGET_COLUMN}' from boolean to integer.")


    # Keep only necessary columns for feature engineering + target + ID
    df_features = df_train[[ID_COLUMN, TEXT_COLUMN, CLEAN_TEXT_COLUMN, TARGET_COLUMN]].copy()


    # --- 2. Create Basic Text Features ---
    df_features = calculate_basic_text_features(df_features, TEXT_COLUMN)
    basic_feature_cols = [col for col in df_features.columns if col.startswith('feat_')]


    # --- 3. Add Sentiment Features ---
    df_features = add_sentiment_features(df_features, TEXT_COLUMN)
    sentiment_feature_cols = [col for col in df_features.columns if col.startswith('feat_sentiment_')]


    # --- 4. Scale Numerical Features (Basic + Sentiment) ---
    # It's often good practice to scale these before combining with TF-IDF,
    # especially for models sensitive to feature scales (like Logistic Regression).
    numerical_features_to_scale = basic_feature_cols + sentiment_feature_cols
    df_features, scaler = scale_numerical_features(df_features, numerical_features_to_scale)
    scaled_feature_cols = numerical_features_to_scale # Keep track of scaled columns


    # --- 5. Create TF-IDF Features ---
    tfidf_params = {
        'max_features': TFIDF_MAX_FEATURES,
        'ngram_range': TFIDF_NGRAM_RANGE,
        'min_df': TFIDF_MIN_DF,
        'max_df': TFIDF_MAX_DF
    }
    df_tfidf, tfidf_vectorizer = create_tfidf_features(df_features, CLEAN_TEXT_COLUMN, **tfidf_params)
    tfidf_feature_cols = df_tfidf.columns.tolist()


    # --- 6. Combine All Features ---
    logging.info("Combining all engineered features...")

    # Select the non-text columns (ID, target, basic/sentiment features)
    df_base = df_features[[ID_COLUMN, TARGET_COLUMN] + scaled_feature_cols]

    # Reset index to ensure alignment during concatenation
    df_base = df_base.reset_index(drop=True)
    df_tfidf = df_tfidf.reset_index(drop=True)

    # Concatenate base features with TF-IDF features
    df_final_features = pd.concat([df_base, df_tfidf], axis=1)

    # Verify the shape and check for NaNs introduced during merging (shouldn't happen with reset_index)
    logging.info(f"Final combined feature matrix shape: {df_final_features.shape}")
    if df_final_features.isnull().any().any():
        logging.warning("NaN values found after combining features. Investigate merging process.")
        # Optional: Fill NaNs if necessary, though ideally the merge should be clean
        # df_final_features.fillna(0, inplace=True) # Example: fill with 0

    all_feature_names = scaled_feature_cols + tfidf_feature_cols
    logging.info(f"Total number of features generated: {len(all_feature_names)}")


    # --- 7. Save Results ---
    logging.info(f"Saving engineered features and objects to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

    # Define output file paths
    features_output_path = OUTPUT_DIR / "engineered_features_train.parquet" # Using Parquet for efficiency
    vectorizer_output_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"
    scaler_output_path = OUTPUT_DIR / "numerical_scaler.joblib"
    feature_names_output_path = OUTPUT_DIR / "feature_names.json"

    try:
        # Save the final DataFrame
        # Using parquet as it's generally more efficient for large DataFrames than CSV
        df_final_features.to_parquet(features_output_path, index=False)
        logging.info(f"Engineered features saved to: {features_output_path}")

        # Save the TF-IDF vectorizer
        joblib.dump(tfidf_vectorizer, vectorizer_output_path)
        logging.info(f"TF-IDF vectorizer saved to: {vectorizer_output_path}")

        # Save the scaler
        joblib.dump(scaler, scaler_output_path)
        logging.info(f"Numerical scaler saved to: {scaler_output_path}")

        # Save the list of final feature names (excluding ID and target)
        import json
        with open(feature_names_output_path, 'w') as f:
            json.dump(all_feature_names, f)
        logging.info(f"List of feature names saved to: {feature_names_output_path}")

    except Exception as e:
        logging.error(f"Error saving results: {e}")

    logging.info("--- Feature Engineering Script Finished ---")


if __name__ == "__main__":
    # Ensure VADER lexicon is downloaded (needed on first run or in new envs)
    try:
        analyzer_check = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        logging.info("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
        logging.info("VADER lexicon downloaded.")

    main()