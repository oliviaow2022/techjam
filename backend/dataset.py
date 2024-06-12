from flask import Blueprint, request, jsonify
from models import db, Dataset, Project

dataset_routes = Blueprint('dataset', __name__)

@dataset_routes.route('/create', methods=['POST'])
def create_dataset():
    data = request.get_json()
    if not data or not 'name' in data or not 'project_id' in data:
        return jsonify({"error": "Bad Request", "message": "Name and project_id are required"}), 400

    project = Project.query.get(data['project_id'])
    if not project:
        return jsonify({"error": "Not Found", "message": "Project not found"}), 404

    dataset = Dataset(name=data['name'], project_id=project.id)
    db.session.add(dataset)
    db.session.commit()

    return jsonify({"message": "Dataset created successfully", "dataset": {"name": dataset.name, "project_id": dataset.project_id}}), 201

