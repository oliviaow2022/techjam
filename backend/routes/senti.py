from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, DataInstance
from flasgger import swag_from
from tempfile import TemporaryDirectory
from services.dataset import get_dataframe
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np
import os
import pickle
from services.S3ImageDataset import s3

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

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(df['data']).toarray()
    y_train = np.array(df['labels'])

    print(df.shape)
    print(df.head())
    print(X_train)
    print(y_train)

    if model_name == "Support Vector Classifier":
        model = SVC(kernel='linear', probability=True)

    # train the estimator model 
    learner = ActiveLearner(
        estimator=model,
        X_training=X_train,
        y_training=y_train,  
        query_strategy=uncertainty_sampling
    )

    with TemporaryDirectory() as tempdir:
        # upload active learner to s3
        local_learner_path = os.path.join(tempdir, f'{model_db.name}.pkl')
        with open(local_learner_path,'wb') as f:
            pickle.dump(learner,f)

        learner_path = f'{project.prefix}/{model_db.name}.pkl'
        s3.upload_file(local_learner_path, project.bucket, learner_path)

        # upload vectorizer to s3
        local_vectorizer_path = os.path.join(tempdir, 'vectorizer.pkl')
        with open(local_vectorizer_path,'wb') as f:
            pickle.dump(vectorizer,f)

        vectorizer_path = f'{project.prefix}/vectorizer.pkl'
        s3.upload_file(local_vectorizer_path, project.bucket, vectorizer_path)

        model_db.saved = learner_path
        db.session.add(model_db)
        db.session.commit()
        print('model saved to', learner_path)

    return jsonify({"message": "Model trained successfully"}), 200


@senti_routes.route('<int:project_id>/query', methods=['POST'])
def query_model(project_id):
    batch_size = request.json.get('batch_size')

    if not batch_size:
        batch_size = 20

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    model_db = Model.query.filter_by(project_id=project.id).first()

    if not model_db:
        return jsonify({"error": "No trained model found"}), 400

    if not model_db.saved:
        data_instances =  DataInstance.query.filter_by(dataset_id=dataset.id, manually_processed=False).limit(batch_size).all()
        data_list = [instance.to_dict() for instance in data_instances]
        return jsonify(data_list), 200

    # load saved model from s3
    response = s3.get_object(Bucket=project.bucket, Key=f'{project.prefix}/{model_db.name}.pkl')
    pickle_data = response['Body'].read()
    learner = pickle.loads(pickle_data)

    # load saved vectorizer from s3
    response = s3.get_object(Bucket=project.bucket, Key=f'{project.prefix}/vectorizer.pkl')
    pickle_data = response['Body'].read()
    vectorizer = pickle.loads(pickle_data)

    df = get_dataframe(dataset.id, return_labelled=False)

    print(df.shape)
    print(df.head())

    X = vectorizer.transform(df['data']).toarray()

    query_idx, _ = learner.query(X, n_instances=batch_size)
    queried_data = df.iloc[query_idx]

    return queried_data.to_json(orient='records'), 200


@senti_routes.route('<int:data_instance_id>/label', methods=['POST'])
def label_data(data_instance_id):
    data_instance = DataInstance.query.get_or_404(data_instance_id)
    dataset = Dataset.query.get_or_404(data_instance.dataset_id)
    project = Project.query.get_or_404(dataset.project_id)
    model_db = Model.query.filter_by(project_id=project.id).first()

    new_label = request.json.get('label')
    if not new_label:
        return jsonify({"error": "New label required"}), 400

    # load saved model from s3
    response = s3.get_object(Bucket=project.bucket, Key=f'{project.prefix}/{model_db.name}.pkl')
    pickle_data = response['Body'].read()
    learner = pickle.loads(pickle_data)

    # load saved vectorizer from s3
    response = s3.get_object(Bucket=project.bucket, Key=f'{project.prefix}/vectorizer.pkl')
    pickle_data = response['Body'].read()
    vectorizer = pickle.loads(pickle_data)
    
    # Update the learner
    X = vectorizer.transform([data_instance.data])
    learner.teach(X, [new_label])

    data_instance.labels = new_label
    data_instance.manually_processed = True
    db.session.commit()
    
    return jsonify({"message": "Label added successfully", 'data_instance_id': data_instance_id}), 200