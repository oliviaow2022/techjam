import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Review Text'] = df['Review Text'].fillna('')
    df = df.dropna(subset=['predicted_label'])
    return df

# Train the selected model
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

# Evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()

# Save the model
def save_model(model, vectorizer, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

# Main function
def main():
    file_path = input("Enter the path to your CSV file: ")
    df = load_dataset(file_path)

    X_train, X_test, y_train, y_test = train_test_split(df['Review Text'], df['predicted_label'], test_size=0.2, random_state=42)

    model_name = input("Enter the model you would like to use (svm, naive_bayes, random_forest, xgboost): ").lower()
    model, vectorizer = train_model(model_name, X_train, y_train)

    report_df = evaluate_model(model, vectorizer, X_test, y_test)
    print("Classification Report:")
    print(report_df)

    save_choice = input("Would you like to save the model? (yes/no): ").lower()
    if save_choice == 'yes':
        model_file_path = input("Enter the file path to save the model (e.g., model.pkl): ")
        save_model(model, vectorizer, model_file_path)
        print(f"Model saved to {model_file_path}")

if __name__ == "__main__":
    main()
