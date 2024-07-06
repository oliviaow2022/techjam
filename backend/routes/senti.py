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

senti_routes = Blueprint('senti', __name__)

global df, learner, vectorizer

@senti_routes.route('<int:dataset_id>/upload', methods=['POST'])
def upload_file(dataset_id):
    text_column = request.json.get("text_column")
    label_column = request.json.get("label_column")
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
        if label_column:
            data_instance.labels = row[label_column]
            data_instance.manually_processed = True
        db.session.add(DataInstance)
        db.session.commit()
    return jsonify({"message": "File uploaded successfully"}), 200


@senti_routes.route('/train', methods=['POST'])
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


@senti_routes.route('/query', methods=['GET'])
def query_model():
    global df, learner, vectorizer

    X = vectorizer.transform(df['text']).toarray()
    query_idx, query_instance = learner.query(X, n_instances=5)
    return jsonify(query_idx.tolist())

@senti_routes.route('<int:data_instance_id>/label', methods=['POST'])
def label_data_instance(data_instance_id):
    data_instance_id = request.json.get('data_instance_id')
    label = request.json.get('label')

    data_instance = DataInstance.query.get_or_404(data_instance_id, description="DataInstance ID not found")
    data_instance.label = label
    db.session.commit()

    return jsonify({"message": "Label updated successfully", "data_instance": {"id": data_instance.id, "label": data_instance.label}})
