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
    # 2. Lexical diversity (unique words ratio)
    # 3. Sentiment polarity (-1 to 1)
    # 4. Subjectivity (0 to 1)
    
    tqdm.pandas(desc="Calculating metrics")
    
    print("Computing metrics (this might take a few minutes)...")
    
    # Text length (words)
    df['word_count'] = df['cleaned_text'].progress_apply(lambda x: len(str(x).split()))
    
    # Lexical Diversity
    def get_lexical_diversity(text):
        if not isinstance(text, str): return 0.0
        words = text.split()
        if not words: return 0.0
        return len(set(words)) / len(words)
        
    df['lexical_diversity'] = df['cleaned_text'].progress_apply(get_lexical_diversity)
    
    # Sentiment (TextBlob polarity & subjectivity)
    # TextBlob can be slow on huge texts, so we might sample or just run it. 
    # For speed, we'll take the first 50,000 characters to estimate.
    def get_sentiment_subj(text):
        if not isinstance(text, str):
            text = ""
        blob = TextBlob(text[:50000])
        return blob.sentiment.polarity, blob.sentiment.subjectivity
        
    print("Computing Sentiment and Subjectivity...")
    sent_subj = df['cleaned_text'].progress_apply(get_sentiment_subj)
    df['sentiment'] = sent_subj.apply(lambda x: x[0])
    df['subjectivity'] = sent_subj.apply(lambda x: x[1])
    
    # Save the metrics (dropping the huge text column)
    metrics_df = df[['movie_title', 'word_count', 'lexical_diversity', 'sentiment', 'subjectivity']]
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved extended metrics to {output_file}")

if __name__ == "__main__":
    extract_metrics()
