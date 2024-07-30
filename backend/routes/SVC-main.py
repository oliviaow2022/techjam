from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

app = Flask(__name__)

# Global variables
data = None
learner = None
vectorizer = None
labeled_indices = set()  # To keep track of manually labeled instances

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
        data = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_csv(file, sep='\t')
        if 'Text' not in data.columns:
            return jsonify({"error": "File must contain a 'text' column"}), 400
        return jsonify({"message": "File uploaded successfully", "samples": len(data)}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/initialize', methods=['POST'])
def initialize_model():
    global learner, vectorizer, data, labeled_indices
    if data is None:
        return jsonify({"error": "No data uploaded"}), 400

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])
    
    # Initialize the model
    model = SVC(kernel='linear', probability=True)
    
    # Initialize the active learner with a small random subset
    n_initial = 5
    initial_idx = np.random.choice(range(len(data)), size=n_initial, replace=False)
    learner = ActiveLearner(
        estimator=model,
        X_training=X[initial_idx],
        y_training=np.random.randint(-1, 2, n_initial),  
        query_strategy=uncertainty_sampling
    )
    
    labeled_indices.update(initial_idx)
    
    return jsonify({"message": "Model initialized", "initial_samples": n_initial}), 200

@app.route('/active_learning', methods=['GET'])
def active_learning():
    global learner, vectorizer, data, labeled_indices
    if learner is None or vectorizer is None or data is None:
        return jsonify({"error": "Model not initialized or no data available"}), 400
    
    # Get unlabeled data
    unlabeled_idx = [i for i in range(len(data)) if i not in labeled_indices]
    X = vectorizer.transform(data.iloc[unlabeled_idx]['Text'])
    
    # Query the instance to be labeled
    query_idx, _ = learner.query(X)
    
    # call app.db and get the index straight from the db

    global_idx = unlabeled_idx[query_idx[0]]
    
    # Return the instance to be labeled
    return jsonify({str(global_idx): data['Text'].iloc[global_idx]}), 200

@app.route('/label', methods=['POST'])
def label_data():
    global learner, vectorizer, data, labeled_indices
    new_label = request.json.get('label')
    text = request.json.get('Text')
    if new_label is None or text is None:
        return jsonify({"error": "Invalid label data"}), 400
    
    # Find the index of the text in the data
    idx = data[data['Text'] == text].index[0]
    
    # Update the learner
    X = vectorizer.transform([text])
    learner.teach(X, [new_label])
    
    labeled_indices.add(idx)
    
    return jsonify({"message": "Label added successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global learner, vectorizer
    if learner is None or vectorizer is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    X_vec = vectorizer.transform([text])
    prediction = learner.predict(X_vec)[0]
    probability = learner.predict_proba(X_vec)[0].max()
    
    return jsonify({"prediction": int(prediction), "confidence": float(probability)}), 200

@app.route('/export', methods=['GET'])
def export_labels():
    global learner, vectorizer, data, labeled_indices
    if learner is None or vectorizer is None or data is None:
        return jsonify({"error": "Model not initialized or no data available"}), 400
    
    X = vectorizer.transform(data['Text'])
    predictions = learner.predict(X)
    probabilities = learner.predict_proba(X)
    
    result = data.copy()
    result['predicted_label'] = predictions
    result['confidence'] = np.max(probabilities, axis=1)
    result['manually_labeled'] = result.index.isin(labeled_indices)
    
    return jsonify(result.to_dict(orient='records')), 200

if __name__ == '__main__':
    app.run(debug=True)