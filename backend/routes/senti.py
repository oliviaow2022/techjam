import numpy as np
import os
import pickle
import pandas as pd
import torch.nn.functional as F

from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, DataInstance, History
from tempfile import TemporaryDirectory
from services.dataset import get_dataframe
from services.S3ImageDataset import s3

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


senti_routes = Blueprint('senti', __name__)

@senti_routes.route('<int:dataset_id>/upload', methods=['POST'])
def upload_file(dataset_id):
    text_column = request.form.get("text_column")
    file = request.files['file']

    if not all([file, text_column]):
        return jsonify({"error": "Missing fields required"}), 400
    
    # Ensure the file is either a CSV or a TXT
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() not in ['.csv', '.txt']:
        return jsonify({"error": "Invalid file type. Only CSV and TXT files are allowed."}), 400

    # Read the file based on its extension
    if file_extension.lower() == '.csv':
        df = pd.read_csv(file)
    elif file_extension.lower() == '.txt':
        df = pd.read_csv(file, delimiter="\t")

    print(df)

    for index, row in df.iterrows():
        data_instance = DataInstance(
            data = row[text_column],
            dataset_id = dataset_id
        )
        db.session.add(data_instance)
        db.session.commit()

        
    return jsonify({"message": "File uploaded successfully"}), 200


@senti_routes.route('<int:project_id>/train', methods=['POST'])
def train_model(project_id):
    model_name = request.json.get('model_name')
    model_description = request.json.get('model_description')
    train_test_split_ratio = request.json.get('train_test_split', 0.8)
    test_size = 1 - train_test_split_ratio

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    model_db = Model.query.filter_by(project_id=project.id).first()
    if not model_db:
        if not model_name:
            return jsonify({"error": "Model name required"}), 400

        model_db = Model(name=model_name, project_id=project.id, description=model_description)

    if not dataset:
       return jsonify({"error": "Dataset does not exist"}), 400
    
    # get labelled data points from databse
    df = get_dataframe(dataset.id, return_labelled=True)
    
    # initialise dataframe with random labels
    if len(df) == 0:
        df = get_dataframe(dataset.id, return_labelled=False)
        df['labels'] = df['labels'].fillna(df['labels'].apply(lambda x: np.random.randint(0, 3) if pd.isnull(x) else x))

    X_train, X_test, y_train, y_test = train_test_split(df['data'], df['labels'], test_size=test_size, random_state=42)

    # convert text to TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(X_train)

    # instantiate model
    if model_name == 'Support Vector Machine (SVM)':
        model = SVC(kernel='linear', probability=True)
    elif model_name == "Naive Bayes":
        model = MultinomialNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost (Extreme Gradient Boosting)":
        model = XGBClassifier(n_estimators=50)
    else:
        raise ValueError("Invalid model name provided")
    
    # train model
    model.fit(X_train, y_train)
    
    with TemporaryDirectory() as tempdir:
        # upload model to s3
        local_model_path = os.path.join(tempdir, f'{model_db.name}.pkl')
        with open(local_model_path,'wb') as f:
            pickle.dump(model,f)

        model_path = f'{project.prefix}/{model_db.name}.pkl'
        s3.upload_file(local_model_path, project.bucket, model_path)

        # upload vectorizer to s3
        local_vectorizer_path = os.path.join(tempdir, 'vectorizer.pkl')
        with open(local_vectorizer_path,'wb') as f:
            pickle.dump(vectorizer,f)

        vectorizer_path = f'{project.prefix}/vectorizer.pkl'
        s3.upload_file(local_vectorizer_path, project.bucket, vectorizer_path)

        model_db.saved = model_path
        db.session.add(model_db)
        db.session.commit()
        print('model saved to', model_path)

    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")

    # save training results to database
    history = History(accuracy=accuracy, precision=precision, recall=recall, f1=f1, model_id=model_db.id)
    db.session.add(history)
    db.session.commit()

    # run model on unlabelled data
    df = get_dataframe(dataset.id, return_labelled=False)
    X_unlabelled = vectorizer.transform(df['data']).toarray()

    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X_unlabelled)
    elif hasattr(model, "decision_function"):
        confidences = model.decision_function(X_unlabelled)
        confidences = F.softmax(confidences, dim=1)
    else:
        raise ValueError("Model does not support confidence scoring")

    # Define a small epsilon value to avoid log(0)
    epsilon = 1e-10

    # Clip confidences to ensure no values are zero
    confidences = np.clip(confidences, epsilon, 1.0)
    entropies = -np.sum(confidences * np.log2(confidences), axis=1)

    X_data_instance_ids = df['id']
    for index, entropy in enumerate(entropies):
        data_instance = DataInstance.query.get_or_404(X_data_instance_ids[index])
        data_instance.entropy = entropy
        db.session.commit()

    return jsonify({"message": "Model trained successfully"}), 200