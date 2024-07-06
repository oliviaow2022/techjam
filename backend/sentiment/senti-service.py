from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, DataInstance
from services.model import run_training, run_labelling_using_model
from flasgger import swag_from
import threading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np

model_routes = Blueprint('model', __name__)

global df, learner, vectorizer

@model_routes.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        return "File uploaded successfully", 200
    return "Failed to upload file", 400

@model_routes.route('/train', methods=['POST'])
def train_model():
    global df, learner, vectorizer

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text']).toarray()

    initial_idx = np.random.choice(range(X.shape[0]), size=10, replace=False)
    initial_samples = X[initial_idx]
    initial_labels = np.random.randint(0, 2, size=10)  

    learner = ActiveLearner(
        estimator=LogisticRegression(),
        query_strategy=uncertainty_sampling,
        X_training=initial_samples,
        y_training=initial_labels
    )

    return "Model trained successfully", 200

@model_routes.route('/query', methods=['GET'])
def query_model():
    global df, learner, vectorizer

    X = vectorizer.transform(df['text']).toarray()
    query_idx, query_instance = learner.query(X, n_instances=5)
    return jsonify(query_idx.tolist())

@model_routes.route('/label', methods=['POST'])
def label_data_instance():
    data_instance_id = request.json.get('data_instance_id')
    label = request.json.get('label')

    data_instance = DataInstance.query.get_or_404(data_instance_id, description="DataInstance ID not found")
    data_instance.label = label
    db.session.commit()

    return jsonify({"message": "Label updated successfully", "data_instance": {"id": data_instance.id, "label": data_instance.label}})
