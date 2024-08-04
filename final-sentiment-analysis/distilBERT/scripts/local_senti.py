import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle
import torch 
from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel


'''
1. having main function for 
    - enquiry about the file path for the data 
    - asking about text column to user
    - load dataset function

2. 

'''
class DistilBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

model = torch.load('/Users/sriyan/Documents/techjam/final-sentiment-analysis/distilBERT/distilbert-model/full_model.pt')
tokeniser = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_sentiment(model, tokenizer, text):
    model.eval()
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not already
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs, dim=-1)  # Use outputs.logits here
    predicted_class = torch.argmax(probabilities, dim=-1)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_sentiment = sentiment_labels[predicted_class.item()]
    return predicted_sentiment, probabilities.squeeze().tolist()

def ground_predict(df, text_column, model, tokeniser):
    sentiments = []
    probabilities_list = []
    for text in df[text_column]:
        if not isinstance(text, str):
            text = str(text)  # Convert to string if it's not already
        sentiment, probabilities = predict_sentiment(model, tokeniser, text)
        sentiments.append(sentiment)
        probabilities_list.append(probabilities)
    df['predicted_sentiment'] = sentiments
    df['probabilities'] = probabilities_list
    return df



def load_dataset(file_path, text_column):
    df = pd.read_csv(file_path)
    df[text_column] = df[text_column].astype(str)  # Convert text column to strings
    df = ground_predict(df, text_column, model, tokeniser)
    df[text_column] = df[text_column].fillna('')
    df = df.dropna(subset=['predicted_sentiment'])
    return df

def convert_labels_to_integers(labels):
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    return labels.map(label_mapping)


def convert_labels_to_strings(labels):
    inv_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return labels.map(inv_label_mapping)


def train_model(model_name, X_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)

    if model_name == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif model_name == 'naive_bayes':
        model = MultinomialNB()
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=50)
        y_train = convert_labels_to_integers(y_train)
    else:
        raise ValueError("Invalid model name provided")

    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

def get_confidence_scores(model, X_tfidf):
    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X_tfidf)
    elif hasattr(model, "decision_function"):
        confidences = model.decision_function(X_tfidf)
        confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())  # normalize
    else:
        raise ValueError("Model does not support confidence scoring")
    return confidences

def evaluate_model(model, vectorizer, X_test, y_test, model_name):
    X_test_tfidf = vectorizer.transform(X_test)
    if model_name == 'xgboost':
        y_test = convert_labels_to_integers(y_test)
    y_pred = model.predict(X_test_tfidf)
    if model_name == 'xgboost':
        y_test = convert_labels_to_strings(y_test)
        y_pred = pd.Series(y_pred)
        y_pred = convert_labels_to_strings(y_pred)
    report = classification_report(y_test, y_pred, output_dict=True) # try to see if you can get classification report as dataframe
    return pd.DataFrame(report).transpose(), y_pred, X_test_tfidf

def save_model(model, vectorizer, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

def active_learning_loop(model_name, X_train, y_train, X_test, y_test, cycles=3): 
    # see if the cycles can be changed to be more dynamic -> user can input???? a bit risky 
    model, vectorizer = train_model(model_name, X_train, y_train)
    for cycle in range(cycles): 
        print(f"Cycle {cycle + 1}")
        report_df, y_pred, X_test_tfidf = evaluate_model(model, vectorizer, X_test, y_test, model_name)
        print("Classification Report:")
        print(report_df)

        # 5 samples with the lowest confidence scores
        confidences = get_confidence_scores(model, X_test_tfidf)
        if len(confidences.shape) == 1:  # check on this again, appropraite method for multi-class classification
            confidences = confidences.reshape(-1, 1)
        least_confident_indices = confidences.min(axis=1).argsort()[:5]

        # manual labelling by user
        for idx in least_confident_indices:
            print(f"Review Text: {X_test.iloc[idx]}")
            new_label = input("Enter the correct label (negative, neutral, positive): ")
            y_train = y_train._append(pd.Series({X_test.index[idx]: new_label}))
            X_train = X_train._append(pd.Series({X_test.index[idx]: X_test.iloc[idx]}))
        
        # Remove the newly labeled samples from the test set
        X_test = X_test.drop(X_test.index[least_confident_indices])
        y_test = y_test.drop(y_test.index[least_confident_indices])

        # Retrain the model with the new labels
        model, vectorizer = train_model(model_name, X_train, y_train)
    
    return model, vectorizer

# user input
def main():
    file_path = input("Enter the path to your CSV file: ")
    text_column = input("Enter the name of the column containing the text data: ")
    
    df = load_dataset(file_path, text_column)

    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df['predicted_sentiment'], test_size=0.2, random_state=42)

    model_name = input("Enter the model you would like to use (svm, naive_bayes, random_forest, xgboost): ").lower()
    model, vectorizer = active_learning_loop(model_name, X_train, y_train, X_test, y_test, cycles=3)

    report_df, _, _ = evaluate_model(model, vectorizer, X_test, y_test, model_name)
    print("Final Classification Report:")
    print(report_df)

    save_choice = input("Would you like to save the model? (yes/no): ").lower()
    if save_choice == 'yes':
        model_file_path = input("Enter the file path to save the model (e.g., model.pkl): ")
        save_model(model, vectorizer, model_file_path)
        print(f"Model saved to {model_file_path}")

if __name__ == "__main__":
    main()