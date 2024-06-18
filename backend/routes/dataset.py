from flask import Blueprint, request, jsonify
from models import db, Dataset, Project, DataInstance
import pandas as pd
from flasgger import swag_from
import json

dataset_routes = Blueprint('dataset', __name__)

@dataset_routes.route('/create', methods=['POST'])
@swag_from({
    'tags': ['Dataset'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'project_id': {'type': 'integer'},
                    'num_classes': {'type': 'integer'},
                    'class_to_label_mapping': {
                        'type': 'object',
                        '0': {'type': 'string'},
                        'example': {0: 'ants', 1: 'bees'}
                    }
                }
            }
        }
    ]
})
def create_dataset():
    name = request.json.get('name')
    project_id = request.json.get('project_id')
    num_classes = request.json.get('num_classes')
    class_to_label_mapping = request.json.get('class_to_label_mapping')

    if not (project_id or name or num_classes or class_to_label_mapping):
        return jsonify({"error": "Bad Request", "message": "Missing fields"}), 400

    project = Project.query.get_or_404(project_id, description="Project ID not found")

    dataset = Dataset(name=name, project_id=project.id, num_classes=num_classes, class_to_label_mapping=json.dumps(class_to_label_mapping))
    db.session.add(dataset)
    db.session.commit()

    return jsonify(dataset.to_dict()), 201


@swag_from({
    'tags': ['Dataset'],
    'summary': 'Return dataframe of entire dataset',
    'parameters': [
        {
            'in': 'path',
            'name': 'dataset_id',
            'type': 'integer',
            'required': True,
            'description': 'Dataset ID'
        },
    ]
})
@dataset_routes.route('/<int:id>/df', methods=['GET'])
def return_dataframe(id):
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    data_instances = DataInstance.query.filter_by(dataset_id=dataset.id).all()
    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df.to_json(orient='records')


@dataset_routes.route('/<int:id>/batch', methods=['GET'])
@swag_from({
    'tags': ['Dataset'],
    'summary': 'Return a batch of 20 data points with no labels',
    'parameters': [
        {
            'in': 'path',
            'name': 'dataset_id',
            'type': 'integer',
            'required': True,
            'description': 'Dataset ID'
        },
    ]
})
def return_batch(id):
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    data_instances = DataInstance.query.filter(
        DataInstance.dataset_id == dataset.id,
        DataInstance.labels == None,
        DataInstance.manually_processed == False
    ).order_by(DataInstance.confidence.asc()).limit(20).all()
    data_list = [instance.to_dict() for instance in data_instances]
    return jsonify(data_list), 200


# for debugging only
@dataset_routes.route('/all', methods=['GET'])
def get_all_datasets():
    datasets = Dataset.query.all()
    dataset_list = [dataset.to_dict() for dataset in datasets]
    return jsonify(dataset_list), 200