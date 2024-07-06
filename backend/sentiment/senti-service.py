from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

app = Flask(__name__)

global df, learner, vectorizer

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        return "File uploaded successfully", 200
    return "File upload failed, please upload a file of the right type", 400

@app.route('/train', methods=['POST'])
def train_model():
    global df, learner, vectorizer

    # tf-idf vectorize the text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text']).toarray()

    # active learning loop?
    initial_idx = np.random.choice(range(X.shape[0]), size=10, replace=False)
    initial_samples = X[initial_idx]
    initial_labels = np.random.randint(0, 2, size=10)  #
    # might need to change this
    learner = ActiveLearner(
        estimator=LogisticRegression(),
        query_strategy=uncertainty_sampling,
        X_training=initial_samples,
        y_training=initial_labels
    )

    return "Model trained successfully", 200

@app.route('/query', methods=['GET'])
def query_model():
    global df, learner, vectorizer
    X = vectorizer.transform(df['text']).toarray()
    query_idx, query_instance = learner.query(X, n_instances=5)
    return jsonify(query_idx.tolist())

if __name__ == '__main__':
    app.run(debug=True)
