from flask import Blueprint, request, jsonify
from models import db, Dataset, Project, DataInstance
import pandas as pd
from flasgger import swag_from
import json
from werkzeug.utils import secure_filename
import os
import uuid
from S3ImageDataset import s3
from services.dataset import get_dataset_config

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

    num_classes, class_to_label_mapping = get_dataset_config(name, num_classes, class_to_label_mapping)

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
        }
    ]
})
@dataset_routes.route('/<int:id>/df', methods=['GET'])
def return_dataframe(id):
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    data_instances = DataInstance.query.filter_by(dataset_id=dataset.id).all()
    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df.to_json(orient='records')


@dataset_routes.route('/<int:project_id>/batch', methods=['POST'])
@swag_from({
    'tags': ['Dataset'],
    'summary': 'Return a batch of data points with no labels',
    'parameters': [
        {
            'in': 'path',
            'name': 'project_id',
            'type': 'integer',
            'required': True,
            'description': 'Project ID'
        },
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'batch_size': {'type': 'integer'},
                }
            }
        }
    ]
})
def return_batch_for_labelling(project_id):
    batch_size = request.json.get('batch_size') 
    data_status = request.json.get('data_status')

    if not batch_size:
        batch_size = 20

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    data_instances = DataInstance.query.filter(
        DataInstance.dataset_id == dataset.id,
        DataInstance.manually_processed == False
    ).order_by(DataInstance.entropy.desc()).limit(batch_size).all()
    data_list = [instance.to_dict() for instance in data_instances]
    return jsonify(data_list), 200


# for debugging only
@dataset_routes.route('/all', methods=['GET'])
def get_all_datasets():
    datasets = Dataset.query.all()
    dataset_list = [dataset.to_dict() for dataset in datasets]
    return jsonify(dataset_list), 200

@swag_from({
    'tags': ['Dataset'],
    'summary': 'Get dataset info',
    'parameters': [
        {
            'name': 'project_id',
            'in': 'path',
            'required': True,
            'description': 'Project ID',
            'schema': {'type': 'integer'}
        }
    ]
})
@dataset_routes.route('/<int:project_id>', methods=['GET'])
def get_dataset(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    return jsonify(dataset.to_dict()), 200


@dataset_routes.route('/<int:id>/upload', methods=['POST'])
@swag_from({
    'tags': ['Dataset'],
    'summary': 'Upload files to a dataset',
    'description': 'Upload multiple files to a specified dataset and store them in an S3 bucket. The filenames will be generated as UUIDs.',
    'parameters': [
        {
            'name': 'id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'ID of the dataset to which the files will be uploaded'
        },
        {
            'name': 'files[]',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Files to upload',
            'collectionFormat': 'multi'
        }
    ]
})
def upload_files(id):
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    project = Project.query.get_or_404(dataset.project_id, description="Project ID not found")

    files = request.files.getlist('files[]')

    UPLOAD_FOLDER = '/tmp'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    for file in files:
        if file:
            # sanitise filename
            original_filename = secure_filename(file.filename)

            extension = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{extension}" if extension else uuid.uuid4().hex
            local_filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(local_filepath)

            # Upload to S3
            s3_filepath = os.path.join(project.prefix, unique_filename)
            s3.upload_file(local_filepath, project.bucket, s3_filepath)

            # Save to database
            data_instance = DataInstance(data=unique_filename, dataset_id=dataset.id)
            db.session.add(data_instance)

            # Remove file from local storage
            os.remove(local_filepath)

    db.session.commit()

    return jsonify({'message': f'{len(files)} files successfully uploaded into dataset'}), 201