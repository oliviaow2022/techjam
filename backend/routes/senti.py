from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, DataInstance
from flasgger import swag_from
from tempfile import TemporaryDirectory
from services.dataset import get_dataframe
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np
import os
import pickle
from S3ImageDataset import s3


senti_routes = Blueprint('senti', __name__)

global df, learner, vectorizer

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

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    model_db = Model.query.filter_by(project_id=project.id).first()
    if not model_db:
        model_db = Model(name="Logistic Regression", project_id=project.id)

    if not dataset:
       return jsonify({"error": "Dataset does not exist"}), 400
    
    df = get_dataframe(dataset.id, return_labelled=False)

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(df['data']).toarray()

    df['labels'] = df['labels'].fillna(df['labels'].apply(lambda x: np.random.randint(0, 3) if pd.isnull(x) else x))
    y_train = np.array(df['labels'])

    print(df.shape)
    print(df.head())

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with TemporaryDirectory() as tempdir:
        local_file_path = os.path.join(tempdir, f"{model_db.name}.pkl")
        with open(local_file_path,'wb') as f:
            pickle.dump(model,f)

        model_path = f'{project.prefix}/{model_db.name}.pkl'
        s3.upload_file(local_file_path, project.bucket, model_path)
        model_db.saved = model_path
        db.session.add(model_db)
        db.session.commit()
        print('model saved to', model_path)

    return jsonify({"message": "Model trained successfully"}), 200


@senti_routes.route('/query', methods=['GET'])
def query_model():
    global df, learner, vectorizer

    X = vectorizer.transform(df['text']).toarray()
    query_idx, query_instance = learner.query(X, n_instances=5)
    return jsonify(query_idx.tolist())