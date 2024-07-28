import pandas as pd
import json
import requests
import os 
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Review Text'] = df['Review Text'].fillna('')
    df = df.dropna(subset=['predicted_label'])
    return df


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
        model = xgb.XGBClassifier(use_label_encoder=False, n_estimators=50)
    else:
        raise ValueError("Invalid model name provided")

    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

# model evaluation at the relevant points
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose(), y_pred, X_test_tfidf

# confidence scores from the 2nd prediction by the models
def get_confidence_scores(model, X_tfidf):
    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X_tfidf)
    elif hasattr(model, "decision_function"):
        confidences = model.decision_function(X_tfidf)
        confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())  
    else:
        raise ValueError("Model does not support confidence scoring")
    return confidences

# no active learner, only confidence sampling
def active_learning_loop(model_name, X_train, y_train, X_test, y_test, cycles=3):
    model, vectorizer = train_model(model_name, X_train, y_train)
    for cycle in range(cycles):
        print(f"Cycle {cycle + 1}")
        report_df, y_pred, X_test_tfidf = evaluate_model(model, vectorizer, X_test, y_test)
        print("Classification Report:")
        print(report_df)

        # only selection of the 5 least confident samples
        confidences = get_confidence_scores(model, X_test_tfidf)
        if len(confidences.shape) == 1:  # decision_function returns a 1D array for binary classification
            confidences = confidences.reshape(-1, 1)
        least_confident_indices = confidences.min(axis=1).argsort()[:5]

        # manual labelling
        for idx in least_confident_indices:
            print(f"Review Text: {X_test.iloc[idx]}")
            new_label = input("Enter the correct label (negative, neutral, positive): ")
            y_train = y_train._append(pd.Series({X_test.index[idx]: new_label})) # use _append, not append
            X_train = X_train._append(pd.Series({X_test.index[idx]: X_test.iloc[idx]}))
        
        # removing the manually labelled sample from the test set 
        X_test = X_test.drop(X_test.index[least_confident_indices])
        y_test = y_test.drop(y_test.index[least_confident_indices])

        # Retrain the model with the new labels
        model, vectorizer = train_model(model_name, X_train, y_train)
    
    return model, vectorizer

# Save the model
def save_model(model, vectorizer, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)


def main():
    file_path = input("Enter the path to your CSV file: ")
    df = load_dataset(file_path)

    X_train, X_test, y_train, y_test = train_test_split(df['Review Text'], df['predicted_label'], test_size=0.2, random_state=42)

    model_name = input("Enter the model you would like to use (svm, naive_bayes, random_forest, xgboost): ").lower()
    model, vectorizer = active_learning_loop(model_name, X_train, y_train, X_test, y_test, cycles=3)

    report_df, _, _ = evaluate_model(model, vectorizer, X_test, y_test)
    print("Final Classification Report:")
    print(report_df)

    save_choice = input("Would you like to save the model? (yes/no): ").lower()
    if save_choice == 'yes':
        model_file_path = input("Enter the file path to save the model (e.g., model.pkl): ")
        save_model(model, vectorizer, model_file_path)
        print(f"Model saved to {model_file_path}")

if __name__ == "__main__":
    main()

