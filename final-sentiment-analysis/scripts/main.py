import pandas as pd
from transformers import pipeline
import torch

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

df = pd.DataFrame({
    'Review Text': ["I love the new design of your website!", "The service was okay, nothing special.", "I am not happy with the product quality.", None]
})

# DeBERTa
df_deberta = sentiment_analysis(df, 'Review Text', 'microsoft/deberta-large-mnli')
print("Results using DeBERTa:")
print(df_deberta)

# BART
df_bart = sentiment_analysis(df, 'Review Text', 'facebook/bart-large-mnli')
print("\nResults using BART:")
print(df_bart)

# Ernie
df_ernie = sentiment_analysis(df, 'Review Text', 'MoritzLaurer/ernie-m-large-mnli-xnli')
print("\nResults using ernie:")
print(df_ernie)