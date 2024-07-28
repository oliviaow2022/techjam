import pandas as pd
from transformers import pipeline
import torch
import os

def sentiment_analysis(df, text_column, model_name):
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=model_name,device=device)
    labels = ["positive", "neutral", "negative"]

    def classify_text(text):
        if pd.isna(text):
            return {"labels": [None], "scores": [None]}
        return classifier(text, labels)

    results = df[text_column].apply(classify_text)
    df['predicted_label'] = results.apply(lambda x: x['labels'][0])
    df['scores'] = results.apply(lambda x: x['scores'][0])

    return df


df = pd.read_csv('/Users/sriyan/Documents/techjam/final-sentiment-analysis/data/train_adjusted.csv')
# df.drop(columns=['Unnamed: 0'], inplace=True)


# DeBERTa
df_deberta = sentiment_analysis(df, 'Review Text', 'microsoft/deberta-large-mnli')
print("labelling with DeBERTa completed! Starting to save file now...")
df_deberta.to_csv('/Users/sriyan/Documents/techjam/final-sentiment-analysis/data/labelled_reviews/deberta.csv', index=False)
# print(df_deberta)

# BART
df_bart = sentiment_analysis(df, 'Review Text', 'facebook/bart-large-mnli')
print("labelling with BART completed! Starting to save file now...")
df_bart.to_csv('/Users/sriyan/Documents/techjam/final-sentiment-analysis/data/labelled_reviews/bart.csv', index=False)
# print(df_bart)

# Ernie
df_ernie = sentiment_analysis(df, 'Review Text', 'MoritzLaurer/ernie-m-large-mnli-xnli')
print("labelling with Ernie completed! Statring to save file now...")
df_ernie.to_csv('/Users/sriyan/Documents/techjam/final-sentiment-analysis/data/labelled_reviews/ernie.csv', index=False)
# print(df_ernie)