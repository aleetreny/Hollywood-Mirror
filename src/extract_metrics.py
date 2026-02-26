import pandas as pd
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob
import numpy as np

def extract_metrics():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    input_file = processed_dir / "movies_cleaned.csv"
    output_file = processed_dir / "movie_metrics.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    print("Loading text data...")
    df = pd.read_csv(input_file)
    
    # We will calculate:
    # 1. Word count
    # 2. Vocabulary size (unique words ratio) - simplified
    # 3. Sentiment polarity (-1 to 1)
    
    tqdm.pandas(desc="Calculating metrics")
    
    print("Computing metrics (this might take a few minutes)...")
    
    # Text length (words)
    df['word_count'] = df['cleaned_text'].progress_apply(lambda x: len(str(x).split()))
    
    # Sentiment (TextBlob polarity)
    # TextBlob can be slow on huge texts, so we might sample or just run it. 
    # For speed, we'll take the first 10,000 characters to estimate sentiment if it's too long, or use the whole text.
    def get_sentiment(text):
        if not isinstance(text, str):
            text = ""
        # Limit to first 50000 chars to avoid massive slowdowns on 2600 full scripts
        return TextBlob(text[:50000]).sentiment.polarity
        
    df['sentiment'] = df['cleaned_text'].progress_apply(get_sentiment)
    
    # Save the metrics (dropping the huge text column)
    metrics_df = df[['movie_title', 'word_count', 'sentiment']]
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved metrics to {output_file}")

if __name__ == "__main__":
    extract_metrics()
