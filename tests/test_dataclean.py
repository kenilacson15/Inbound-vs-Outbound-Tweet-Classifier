import csv
from pathlib import Path
import pandas as pd

# Define file paths using pathlib for improved path management
input_path = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\cleaned\twcs_sample_cleaned.csv")
output_dir = Path(r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\cleaned")
report_path = output_dir / "data_quality_report.txt"

# Expected schema with column names and expected data types
expected_columns = {
    "tweet_id": "int64",
    "author_id": "object",
    "inbound": "bool",
    "created_at": "object",  # Dates will be converted later
    "text": "object",
    "response_tweet_id": "object",  # Some may be missing, thus object type
    "in_response_to_tweet_id": "float64",  # Missing values present
    "clean_text": "object"
}

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file using pandas.
    
    For a cleaned CSV, we assume comma-separated data. Adjust the 'sep' parameter
    if your file uses a different delimiter. We use engine="python" and skip any bad lines.
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=",",
            engine="python",
            on_bad_lines='skip'
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading data from {file_path}: {e}")

def check_columns(df: pd.DataFrame) -> str:
    """
    Check for expected columns and report any differences.
    """
    report = []
    actual_columns = list(df.columns)
    report.append("Expected columns:\n" + str(list(expected_columns.keys())) + "\n")
    report.append("Actual columns:\n" + str(actual_columns) + "\n")
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        report.append("Missing expected columns: " + str(missing_cols) + "\n")
    else:
        report.append("All expected columns are present.\n")
    
    return "\n".join(report)

def check_data_types(df: pd.DataFrame) -> str:
    """
    Check and report the data types of each expected column.
    """
    report = ["Data Types:"]
    for col, exp_type in expected_columns.items():
        if col in df.columns:
            actual_type = df[col].dtype
            report.append(f" - {col}: Expected {exp_type} | Actual {actual_type}")
        else:
            report.append(f" - {col}: Column not found.")
    return "\n".join(report)

def check_missing_values(df: pd.DataFrame) -> str:
    """
    Report the count of missing values for each column.
    """
    report = ["Missing Values per Column:"]
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        report.append(f" - {col}: {count}")
    return "\n".join(report)

def check_duplicates(df: pd.DataFrame) -> str:
    """
    Check for duplicate rows in the DataFrame.
    """
    duplicated_count = df.duplicated().sum()
    return f"Total duplicate rows: {duplicated_count}\n"

def check_created_at(df: pd.DataFrame) -> str:
    """
    Convert the 'created_at' column to datetime and report any conversion issues.
    """
    report = ["Created_at Date Conversion Check:"]
    try:
        df["created_at_converted"] = pd.to_datetime(df["created_at"], errors="coerce")
        null_dates = df["created_at_converted"].isnull().sum()
        if null_dates > 0:
            report.append(f" - Warning: {null_dates} rows could not be parsed as dates.")
        else:
            report.append(" - All date values successfully converted.")
    except Exception as e:
        report.append(f"Error during date conversion: {e}")
    return "\n".join(report)

def check_text_integrity(df: pd.DataFrame) -> str:
    """
    Validate the 'text' column by ensuring no entries are empty after stripping whitespace.
    """
    report = ["Text Column Integrity:"]
    if "text" in df.columns:
        empty_texts = df["text"].astype(str).apply(lambda x: len(x.strip()) == 0).sum()
        if empty_texts:
            report.append(f" - Found {empty_texts} empty text entries in 'text' column.")
        else:
            report.append(" - All entries in 'text' column are non-empty.")
    else:
        report.append(" - Column 'text' is missing; cannot check text integrity.")
    return "\n".join(report)

def main():
    try:
        # Load data
        df = load_data(input_path)
        report_lines = [
            "DATA QUALITY REPORT",
            "-" * 50,
            f"Dataset shape: {df.shape}\n"
        ]
        
        # Append report sections for each check
        report_lines.append(check_columns(df))
        report_lines.append(check_data_types(df))
        report_lines.append(check_missing_values(df))
        report_lines.append(check_duplicates(df))
        report_lines.append(check_created_at(df))
        report_lines.append(check_text_integrity(df))
        
        # Optional: Preview the first few rows of data
        report_lines.append("Sample Data (first 5 rows):")
        report_lines.append(df.head().to_string())
        
        # Join report lines and write to report file
        report_content = "\n".join(report_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print("Data quality check completed. Report saved to:")
        print(report_path)
    except Exception as e:
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
