import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# -------------------------------
# Configurations & Paths
# -------------------------------

# Define paths to dataset and output directory
DATASET_PATH = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\cleaned\twcs_sample_cleaned.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\EDA"

# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -------------------------------
# Load the dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Display the first few rows of the dataset for a quick sanity check
print("First few rows of the dataset:")
print(df.head())

# -------------------------------
# Basic Data Inspection and Overview
# -------------------------------
print("\nDataset Information:")
buffer = []
buffer.append("Dataset Information:\n")
buffer.append(df.info(buf=open(os.devnull, 'w')))
buffer.append("\nMissing Values Per Column:\n")
missing_values = df.isnull().sum()
buffer.append(str(missing_values))
buffer.append("\nSummary Statistics for Numerical Columns:\n")
buffer.append(str(df.describe()))

# Save the basic information to a text file in the output directory
with open(os.path.join(OUTPUT_DIR, "dataset_overview.txt"), "w", encoding="utf-8") as f:
    f.write("Dataset Head:\n")
    f.write(df.head().to_string())
    f.write("\n\nMissing Values Per Column:\n")
    f.write(missing_values.to_string())
    f.write("\n\nSummary Statistics for Numerical Columns:\n")
    f.write(df.describe().to_string())
print("Basic dataset overview saved to 'dataset_overview.txt'.")

# -------------------------------
# Distribution of Categorical Variable: 'inbound'
# -------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='inbound')
plt.title("Distribution of 'inbound' Tweets")
plt.xlabel("Inbound")
plt.ylabel("Count")
inbound_plot_path = os.path.join(OUTPUT_DIR, "inbound_distribution.png")
plt.savefig(inbound_plot_path, bbox_inches='tight')
plt.close()
print(f"Inbound count plot saved to {inbound_plot_path}")

# -------------------------------
# Text Length Distribution Analysis
# -------------------------------
def add_text_length(df, col):
    # Create a new column with the length of each tweet's text
    new_col = f"{col}_length"
    df[new_col] = df[col].astype(str).apply(len)
    return new_col

text_length_col = add_text_length(df, "text")
clean_text_length_col = add_text_length(df, "clean_text")

plt.figure(figsize=(10, 5))
sns.histplot(df[text_length_col], bins=30, kde=True, color='blue', label='Text Length')
sns.histplot(df[clean_text_length_col], bins=30, kde=True, color='green', label='Clean Text Length', alpha=0.6)
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.title("Distribution of Tweet Text Lengths")
plt.legend()
text_length_plot_path = os.path.join(OUTPUT_DIR, "text_length_distribution.png")
plt.savefig(text_length_plot_path, bbox_inches='tight')
plt.close()
print(f"Text length distribution plot saved to {text_length_plot_path}")

# -------------------------------
# Word Cloud Generation
# -------------------------------
def generate_wordcloud(text, title, output_path):
    stop_words = set(stopwords.words("english"))
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words)
    wc.generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to {output_path}")

# Create a combined text string from 'text' and 'clean_text' columns respectively
combined_text = " ".join(df["text"].astype(str).tolist())
combined_clean_text = " ".join(df["clean_text"].astype(str).tolist())

# Generate word clouds for both text columns
wc_text_path = os.path.join(OUTPUT_DIR, "wordcloud_text.png")
wc_clean_text_path = os.path.join(OUTPUT_DIR, "wordcloud_clean_text.png")
generate_wordcloud(combined_text, "Word Cloud - Original Text", wc_text_path)
generate_wordcloud(combined_clean_text, "Word Cloud - Cleaned Text", wc_clean_text_path)

# -------------------------------
# Additional EDA: Frequent Words Analysis
# -------------------------------
def plot_top_words(text_series, title, output_path, top_n=20):
    stop_words = set(stopwords.words("english"))
    words = " ".join(text_series.astype(str).tolist()).lower().split()
    # Filter out stopwords and non-alphabetic tokens
    words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(words)
    common_words = word_counts.most_common(top_n)
    
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Top words plot saved to {output_path}")

top_words_plot_path = os.path.join(OUTPUT_DIR, "top_words_original_text.png")
plot_top_words(df["text"], "Top 20 Most Frequent Words (Original Text)", top_words_plot_path)

top_words_clean_plot_path = os.path.join(OUTPUT_DIR, "top_words_clean_text.png")
plot_top_words(df["clean_text"], "Top 20 Most Frequent Words (Cleaned Text)", top_words_clean_plot_path)

# -------------------------------
# Correlation Analysis (if applicable)
# -------------------------------
# If you have any numeric columns aside from the text length columns, you can compute correlations.
# Here we add the text length columns and compute a correlation heatmap.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Include our derived text length columns if not already numeric
if text_length_col not in numeric_cols:
    numeric_cols.append(text_length_col)
if clean_text_length_col not in numeric_cols:
    numeric_cols.append(clean_text_length_col)

if len(numeric_cols) > 1:
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    corr_plot_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {corr_plot_path}")
else:
    print("Not enough numeric columns to compute a correlation heatmap.")

print("Deep EDA completed. All outputs are saved in:", OUTPUT_DIR)
