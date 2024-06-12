from flask import Blueprint, request, jsonify
from models import db, DataInstance, Dataset

data_instance_routes = Blueprint('data_instance', __name__)

@data_instance_routes.route('/create', methods=['POST'])
def create_data_instance():
    data = request.get_json()
    if not data or not 'data' in data or not 'labels' in data or not 'dataset_id' in data:
        return jsonify({"error": "Bad Request", "message": "Data, labels, and dataset_id are required"}), 400

    dataset = Dataset.query.get(data['dataset_id'])
    if not dataset:
        return jsonify({"error": "Not Found", "message": "Dataset not found"}), 404

    data_instance = DataInstance(data=data['data'], labels=data['labels'], dataset_id=dataset.id)
    db.session.add(data_instance)
    db.session.commit()

    return jsonify({"message": "DataInstance created successfully", "data_instance": {"id": data_instance.id, "data": data_instance.data, "labels": data_instance.labels, "dataset_id": data_instance.dataset_id}}), 201
