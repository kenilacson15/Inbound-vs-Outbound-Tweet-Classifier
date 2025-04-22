import pandas as pd

# Define file paths
raw_data_path = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\raw\customer-support\twcs\twcs.csv"
output_path = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\twcs_sample.csv"

# Parameters for sampling
sample_size = 50000
random_seed = 42  # For reproducibility

def sample_data(raw_path: str, output_path: str, n_samples: int, seed: int) -> None:
    """
    Reads the raw CSV data, samples n_samples rows randomly, and writes the sample to output_path.
    
    Parameters:
        raw_path (str): The file path to the raw CSV data.
        output_path (str): The file path where the sampled data should be saved.
        n_samples (int): The number of samples to extract.
        seed (int): Random seed for reproducibility.
    """
    print("Reading the raw data...")
    # Load the dataset into a DataFrame
    df = pd.read_csv(raw_path)
    
    total_samples = len(df)
    print(f"Total samples in the raw dataset: {total_samples}")
    
    # Check if sample size exceeds available data
    if total_samples < n_samples:
        print("Warning: Requested sample size is larger than the dataset size. Using full dataset.")
        sample_df = df
    else:
        print(f"Sampling {n_samples} data points from the dataset...")
        sample_df = df.sample(n=n_samples, random_state=seed)
    
    # Save the sampled data to a new CSV file
    sample_df.to_csv(output_path, index=False)
    print(f"Sampled dataset with {len(sample_df)} records saved to {output_path}")

if __name__ == "__main__":
    sample_data(raw_data_path, output_path, sample_size, random_seed)
