import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
analyzer = SentimentIntensityAnalyzer()

# ✅ Use raw string to avoid unicodeescape issues
file_path ="C:\Users\HP\OneDrive\Desktop\NM Project\test.csv"# Ensure the filename is correct

if not os.path.exists(file_path):
    print(f"File not found at path: {file_path}")
else:
    df = pd.read_csv(file_path, encoding='latin1')

    if 'text' not in df.columns:
        print("The CSV file must contain a column named 'text'.")
    else:
        df['Sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

        def classify_sentiment(score):
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        df['Sentiment_Label'] = df['Sentiment'].apply(classify_sentiment)

        print(df[['text', 'Sentiment', 'Sentiment_Label']].head())

        sentiment_counts = df['Sentiment_Label'].value_counts()

        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Sentiment Analysis of Social Media Conversations')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Conversations')
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
