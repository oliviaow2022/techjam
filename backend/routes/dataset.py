from flask import Blueprint, request, jsonify
from models import db, Dataset, Project, DataInstance
import pandas as pd

dataset_routes = Blueprint('dataset', __name__)

@dataset_routes.route('/create', methods=['POST'])
def create_dataset():
    project_id = request.json.get('project_id')
    name = request.json.get('name')

    if not (project_id or name):
        return jsonify({"error": "Bad Request", "message": "Name and project_id are required"}), 400

    project = Project.query.get_or_404(project_id, description="Project ID not found")

    dataset = Dataset(name=name, project_id=project.id)
    db.session.add(dataset)
    db.session.commit()

    return jsonify(dataset.to_dict()), 201


@dataset_routes.route('/<int:id>/df', methods=['GET'])
def return_dataframe(id):
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    data_instances = DataInstance.query.filter_by(dataset_id=dataset.id).all()
    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df.to_json(orient='records')


@dataset_routes.route('/<int:id>/batch', methods=['GET'])
def return_batch():
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    data_instances = DataInstance.query.filter_by(dataset_id=dataset.id, manually_processed=False).order_by(DataInstance.confidence.asc()).limit(20).all()
    data_list = [instance.to_dict() for instance in data_instances]
    return jsonify(data_list), 200


# for debugging only
@dataset_routes.route('/all', methods=['GET'])
def get_all_datasets():
    datasets = Dataset.query.all()
    dataset_list = [dataset.to_dict() for dataset in datasets]
    return jsonify(dataset_list), 200