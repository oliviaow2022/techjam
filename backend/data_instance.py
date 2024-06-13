from flask import Blueprint, request, jsonify
from models import db, DataInstance, Dataset

data_instance_routes = Blueprint('data_instance', __name__)

@data_instance_routes.route('/create', methods=['POST'])
def create_data_instance():
    data = request.get_json()
    if not data or not 'data' in data or not 'dataset_id' in data:
        return jsonify({"error": "Bad Request", "message": "Data, labels, and dataset_id are required"}), 400

    dataset = Dataset.query.get(data['dataset_id'])
    if not dataset:
        return jsonify({"error": "Not Found", "message": "Dataset not found"}), 404

    data_instance = DataInstance(data=data['data'], dataset_id=dataset.id)
    db.session.add(data_instance)
    db.session.commit()

    return jsonify({"message": "DataInstance created successfully", "data_instance": {"id": data_instance.id, "data": data_instance.data, "dataset_id": data_instance.dataset_id}}), 201

@app.route('/data_instance/<int:id>/set_label', methods=['POST'])
def set_label(id):
    # Get the data instance by ID
    data_instance = DataInstance.query.get_or_404(id)
    
    # Get the new labels from the request
    if not request.json or 'labels' not in request.json:
        return jsonify({'error': 'Invalid input'}), 400
    
    new_labels = request.json['labels']

    if not isinstance(new_labels, list):
        return jsonify({'error': 'Labels must be a list'}), 400

    # Convert all elements to strings
    new_labels = [str(label) for label in new_labels]
    
    # Update the labels
    data_instance.labels = new_labels
    
    # Commit the changes to the database
    db.session.commit()
    
    return jsonify({'message': 'Labels updated successfully', 'data_instance': data_instance.id}), 200