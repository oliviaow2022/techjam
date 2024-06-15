from flask import Blueprint, request, jsonify
from models import db, DataInstance, Dataset

data_instance_routes = Blueprint('data_instance', __name__)

@data_instance_routes.route('/create', methods=['POST'])
def create_data_instance():
    data = request.json.get('data')
    dataset_id = request.json.get('dataset_id')

    if not (data or dataset_id):
        return jsonify({"error": "Bad Request", "message": "Data, and dataset_id are required"}), 400

    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")

    data_instance = DataInstance(data=data, dataset_id=dataset.id)
    db.session.add(data_instance)
    db.session.commit()

    return jsonify({"message": "DataInstance created successfully", "data_instance": {"id": data_instance.id, "data": data_instance.data, "dataset_id": data_instance.dataset_id}}), 201


@data_instance_routes.route('/<int:id>/set_label', methods=['POST'])
def set_label(id):
    # Get the data instance by ID
    data_instance = DataInstance.query.get_or_404(id, description="Data instance ID not found")
    
    new_labels = request.json.get('labels')

    # Get the new labels from the request
    if not new_labels:
        return jsonify({'error': 'Invalid input'}), 400

    if not isinstance(new_labels, str):
        new_labels = str(new_labels)
    
    # Update the labels
    data_instance.labels = new_labels
    data_instance.manually_processed = True
    
    # Commit the changes to the database
    db.session.commit()
    
    return jsonify({'message': 'Labels updated successfully', 'data_instance': data_instance.id}), 200

