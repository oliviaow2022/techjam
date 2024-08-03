import numpy as np
import os
import pickle
import pandas as pd
import torch.nn.functional as F

from celery import shared_task
from models import db, Model, History, DataInstance
from services.dataset import get_dataframe
from services.S3ImageDataset import s3
from tempfile import TemporaryDirectory

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


@shared_task()
def run_training(project_dict, model_dict, dataset_dict, TEST_SIZE):

    # get labelled data points from databse
    df = get_dataframe(dataset_dict['id'], return_labelled=True)
    
    # initialise dataframe with random labels
    if len(df) == 0:
        df = get_dataframe(dataset_dict['id'], return_labelled=False)
        df['labels'] = df['labels'].fillna(df['labels'].apply(lambda x: np.random.randint(0, 3) if pd.isnull(x) else x))

    X_train, X_test, y_train, y_test = train_test_split(df['data'], df['labels'], test_size=TEST_SIZE, random_state=42)

    # convert text to TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(X_train)

    # instantiate model
    if model_dict['name'] == 'Support Vector Machine (SVM)':
        model = SVC(kernel='linear', probability=True)
    elif model_dict['name'] == "Naive Bayes":
        model = MultinomialNB()
    elif model_dict['name'] == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_dict['name'] == "XGBoost (Extreme Gradient Boosting)":
        model = XGBClassifier(n_estimators=50)
    else:
        raise ValueError("Invalid model name provided")
    
    # train model
    model.fit(X_train, y_train)
    
    with TemporaryDirectory() as tempdir:
        # upload model to s3
        local_model_path = os.path.join(tempdir, f"{model_dict['name']}_{model_dict['id']}.pkl")
        with open(local_model_path,'wb') as f:
            pickle.dump(model,f)

        model_path = f"{project_dict['prefix']}/{model_dict['name']}_{model_dict['id']}.pkl"
        s3.upload_file(local_model_path, project_dict['bucket'], model_path)

        # upload vectorizer to s3
        local_vectorizer_path = os.path.join(tempdir, f"{model_dict['name']}_{model_dict['id']}_vectorizer.pkl")
        with open(local_vectorizer_path,'wb') as f:
            pickle.dump(vectorizer,f)

        vectorizer_path = f"{project_dict['prefix']}/{model_dict['name']}_{model_dict['id']}_vectorizer.pkl"
        s3.upload_file(local_vectorizer_path, project_dict['bucket'], vectorizer_path)

        model_db = Model.query.get_or_404(model_dict['id'])
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
    df = get_dataframe(dataset_dict['id'], return_labelled=False)
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

    X_data_instance_ids = df['id'].tolist()

    for index, entropy in enumerate(entropies):
        data_instance = DataInstance.query.get_or_404(X_data_instance_ids[index])
        data_instance.entropy = entropy
        db.session.commit()
