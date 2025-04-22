import os
import re
import string
import logging
import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are available; if not, download them.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# NLP Preprocessing Resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_data(input_filepath: str) -> pd.DataFrame:
    """
    Load the dataset from the CSV file, with fallback for nonstandard formatting.
    """
    try:
        df = pd.read_csv(input_filepath, sep=",", engine="python")
        # If the CSV is loaded as a single column, split based on comma delimiters.
        if df.shape[1] == 1 and ("," in df.columns[0]):
            logging.info("Single-column CSV detected; attempting to split header columns.")
            new_cols = df.columns[0].split(',')
            df = pd.read_csv(input_filepath, sep=",", header=0, names=new_cols, engine="python")
        logging.info("Data loaded successfully with %d columns.", len(df.columns))
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


def perform_eda(df: pd.DataFrame, expected_columns: list) -> None:
    """
    Perform basic exploratory data analysis and log the results.
    """
    logging.info("### Basic Dataset Information ###")
    logging.info("Shape: %s", df.shape)
    logging.info("Columns and Data Types:\n%s", df.dtypes)
    logging.info("Missing Values Per Column:\n%s", df.isna().sum())
    logging.info("Descriptive Statistics (Numeric):\n%s", df.describe())
    logging.info("Descriptive Statistics (Non-Numeric):\n%s", df.describe(include=['object']))

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logging.warning("Missing expected columns: %s", missing)
    else:
        logging.info("All expected columns are present.")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing values for selected columns.
    """
    if 'response_tweet_id' in df.columns:
        df['response_tweet_id'] = df['response_tweet_id'].fillna("")
    if 'in_response_to_tweet_id' in df.columns:
        # Optionally, one could choose a different strategy; here we keep it as NaN.
        df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(np.nan)
    return df


def clean_text(text: str) -> str:
    """
    Clean and normalize tweet text for NLP analysis.
    """
    # Ensure input is string
    if not isinstance(text, str):
        return ""
    
    # Remove newline characters and extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions (@username) and hashtags (#hashtag)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation using translate method
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text by whitespace
    tokens = text.split()
    
    # Filter out stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Return the cleaned text
    return " ".join(tokens)


def preprocess_text_column(df: pd.DataFrame, source_col: str = 'text', target_col: str = 'clean_text') -> pd.DataFrame:
    """
    Apply text cleaning to the source column and save the result in the target column.
    """
    if source_col not in df.columns:
        logging.error("Source column '%s' not found in DataFrame.", source_col)
        raise ValueError(f"Column '{source_col}' does not exist in DataFrame.")
    
    logging.info("Starting text cleaning process on column '%s'...", source_col)
    df[target_col] = df[source_col].apply(clean_text)
    logging.info("Text cleaning completed for column '%s'.", source_col)
    return df


def save_data(df: pd.DataFrame, output_filepath: str) -> None:
    """
    Save the processed DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_filepath, index=False)
        logging.info("Cleaned data saved to: %s", output_filepath)
    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise


def main(args):
    # Define paths from command-line arguments
    input_filepath = args.input
    output_dir = args.output_dir
    output_filepath = os.path.join(output_dir, "twcs_sample_cleaned.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = load_data(input_filepath)
    
    # Expected columns from data report
    expected_columns = ['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 
                        'response_tweet_id', 'in_response_to_tweet_id', 'clean_text']
    
    # Perform initial EDA
    perform_eda(df, expected_columns)
    
    # Handle missing values in specific columns
    df = handle_missing_values(df)
    
    # Clean the text column and create/update clean_text column
    df = preprocess_text_column(df, source_col='text', target_col='clean_text')
    
    # Save the cleaned data to a new CSV file
    save_data(df, output_filepath)
    
    # Log a sample of the cleaned data for verification
    if set(['tweet_id', 'text', 'clean_text']).issubset(df.columns):
        logging.info("### Sample Cleaned Data ###")
        logging.info("\n%s", df[['tweet_id', 'text', 'clean_text']].head())
    else:
        logging.warning("Some expected columns are missing for final output display.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Preprocessing for Twitter Customer Support Data")
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\twcs_sample.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\cleaned",
        help="Directory where the cleaned CSV file will be saved."
    )
    args = parser.parse_args()
    main(args)
