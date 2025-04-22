import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already available
nltk.download('vader_lexicon')

def load_data(input_path):
    # Load the dataset; adjust parameters if needed (delimiter, encoding, etc.)
    df = pd.read_csv(input_path)
    return df

def basic_summary(df, output_dir):
    # Gather basic information
    summary_lines = []
    summary_lines.append("### Basic Dataset Information ###")
    summary_lines.append(f"Shape: {df.shape}\n")
    summary_lines.append("Columns and Data Types:")
    summary_lines.append(df.dtypes.to_string())
    summary_lines.append("\n\nMissing Values:")
    summary_lines.append(df.isnull().sum().to_string())
    summary_lines.append("\n\nDescriptive Statistics (Numeric):")
    summary_lines.append(df.describe(include="number").to_string())
    summary_lines.append("\n\nDescriptive Statistics (Non-Numeric):")
    summary_lines.append(df.describe(include="object").to_string())

    # Write the summary to a text file
    summary_file = os.path.join(output_dir, "EDA_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines))
    
    # Save a head sample
    df.head().to_csv(os.path.join(output_dir, "head_sample.csv"), index=False)

def preprocess_data(df):
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Create additional time features for further analysis
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    
    # Calculate tweet text length
    df['text_length'] = df['text'].astype(str).apply(len)
    
    return df

def plot_time_series(df, output_dir):
    # Plot tweets over time (daily count)
    plt.figure(figsize=(12, 6))
    tweet_counts = df['date'].value_counts().sort_index()
    tweet_counts.plot(kind='line', marker='o')
    plt.title("Tweets Over Time (Daily Count)")
    plt.xlabel("Date")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tweets_over_time.png"))
    plt.close()

def plot_inbound_counts(df, output_dir):
    # Plot count of inbound vs non-inbound tweets
    plt.figure(figsize=(6, 4))
    sns.countplot(x='inbound', data=df)
    plt.title("Count of Inbound Tweets")
    plt.xlabel("Inbound")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inbound_count.png"))
    plt.close()

def plot_top_authors(df, output_dir):
    # Plot top 10 authors by tweet frequency
    plt.figure(figsize=(10, 6))
    top_authors = df['author_id'].value_counts().head(10)
    sns.barplot(x=top_authors.index.astype(str), y=top_authors.values, palette="viridis")
    plt.title("Top 10 Authors by Number of Tweets")
    plt.xlabel("Author ID")
    plt.ylabel("Tweet Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_authors.png"))
    plt.close()

def plot_text_length_distribution(df, output_dir):
    # Plot the distribution of tweet text lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=30, kde=True, color='teal')
    plt.title("Distribution of Tweet Text Length")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tweet_text_length.png"))
    plt.close()

def plot_hourly_distribution(df, output_dir):
    # Plot the distribution of tweets by hour of the day
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour', data=df, palette="coolwarm")
    plt.title("Distribution of Tweets by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tweets_by_hour.png"))
    plt.close()

def sentiment_analysis(df, output_dir):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Compute sentiment scores for each tweet
    df['sentiment_score'] = df['text'].astype(str).apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Plot distribution of sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment_score'], bins=30, kde=True, color='coral')
    plt.title("Distribution of Sentiment Scores (Compound)")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
    plt.close()
    
    # Save a sample sentiment analysis result to CSV
    sentiment_sample = df[['tweet_id', 'text', 'sentiment_score']].head(50)
    sentiment_sample.to_csv(os.path.join(output_dir, "sentiment_sample.csv"), index=False)

def generate_wordcloud(df, output_dir):
    # Combine all text from tweets into a single string
    text_combined = " ".join(df['text'].dropna().astype(str).tolist())
    stopwords = set(STOPWORDS)
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stopwords, collocations=False).generate(text_combined)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Tweet Texts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tweet_wordcloud.png"))
    plt.close()

def plot_missing_data_heatmap(df, output_dir):
    # Visualize missing data with a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_data_heatmap.png"))
    plt.close()

def correlation_matrix(df, output_dir):
    # Compute correlation matrix for numeric features
    plt.figure(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix (Numeric Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

def main():
    # File paths
    input_path = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\twcs_sample.csv"
    output_dir = r"C:\Users\Ken Ira Talingting\Desktop\Projects\twitter-customer_support-NLP\data\processed\EDA"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df = load_data(input_path)
    
    # Basic summary and head sample
    basic_summary(df, output_dir)
    
    # Preprocess data (datetime conversion, additional features)
    df = preprocess_data(df)
    
    # Generate plots and analysis outputs
    plot_time_series(df, output_dir)
    plot_inbound_counts(df, output_dir)
    plot_top_authors(df, output_dir)
    plot_text_length_distribution(df, output_dir)
    plot_hourly_distribution(df, output_dir)
    sentiment_analysis(df, output_dir)
    generate_wordcloud(df, output_dir)
    plot_missing_data_heatmap(df, output_dir)
    correlation_matrix(df, output_dir)
    
    print("Comprehensive EDA completed. Check the output directory for results.")

if __name__ == "__main__":
    main()
