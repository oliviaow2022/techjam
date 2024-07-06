from flask import Blueprint, request, jsonify
from models import db, Model, Dataset, Project, DataInstance
import pandas as pd
from flasgger import swag_from
import json
from werkzeug.utils import secure_filename
import os
import uuid
from S3ImageDataset import s3
from services.dataset import get_dataset_config
import zipfile

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

    if not batch_size:
        batch_size = 20

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    print(dataset)

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    data_instances = DataInstance.query.filter(
        DataInstance.dataset_id == dataset.id,
        DataInstance.manually_processed == False
    ).order_by(DataInstance.entropy.desc()).limit(batch_size).all()
    data_list = [instance.to_dict() for instance in data_instances]
    print(data_list)
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
    
    return jsonify({'dataset': dataset.to_dict(), 'project': project.to_dict()}), 200


@dataset_routes.route('/<int:id>/upload', methods=['POST'])
@swag_from({
    'tags': ['Dataset'],
    'summary': 'Upload a zipped folder to a dataset',
    'description': 'Upload a zipped folder to a specified dataset, extract the files, and store them in an S3 bucket. The filenames will be generated as UUIDs.',
    'parameters': [
        {
            'name': 'id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'ID of the dataset to which the files will be uploaded'
        },
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Zipped folder to upload'
        }
    ],
    'responses': {
        '201': {
            'description': 'Files successfully uploaded into dataset'
        },
        '400': {
            'description': 'Invalid request or no file part in the request'
        },
        '404': {
            'description': 'Dataset or Project ID not found'
        }
    }
})
def upload_files(id):
    print(request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    dataset = Dataset.query.get_or_404(id, description="Dataset ID not found")
    project = Project.query.get_or_404(dataset.project_id, description="Project ID not found")

    file = request.files['file']
    print(file.filename)

    UPLOAD_FOLDER = './tmp'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Sanitize filename
    original_filename = secure_filename(file.filename)
    local_zip_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(local_zip_path)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_FOLDER)

    os.remove(local_zip_path)  # Remove the zip file after extraction

    # Process each file in the extracted folder
    for root, _, files in os.walk(UPLOAD_FOLDER):
        for file_name in files:
            if file_name.endswith('.zip'):  # Skip the original zip file
                continue

            local_filepath = os.path.join(root, file_name)
            unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(file_name)[1]}"
            # s3_filepath = os.path.join(project.prefix, unique_filename)
            s3_filepath = f'{project.prefix}/{unique_filename}'
            
            # Upload to S3
            s3.upload_file(local_filepath, os.getenv('S3_BUCKET'), s3_filepath)
            
            # Save to database
            data_instance = DataInstance(data=unique_filename, dataset_id=dataset.id)
            db.session.add(data_instance)

            # Remove file from local storage
            os.remove(local_filepath)

    db.session.commit()

    return jsonify({'message': 'Files successfully uploaded into dataset'}), 201