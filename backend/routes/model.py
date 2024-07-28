from flask import Blueprint, request, jsonify, send_file
from models import db, Model, Project, Dataset
from services.model import run_training, run_labelling_with_model
from flasgger import swag_from
import threading
from S3ImageDataset import s3
from tempfile import TemporaryDirectory
from botocore.exceptions import ClientError
import os

model_routes = Blueprint('model', __name__)

@model_routes.route('/create', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'description': 'model name must be in this list [ResNet-18, DenseNet-121, AlexNet, ConvNext Base]!!',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'project_id': {'type': 'integer'}
                },
                'required': ['name', 'project_id']
            }
        }
    ]
})
def create_model():
    name = request.json.get('name')
    project_id = request.json.get('project_id')

    if not (name or project_id):
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    model = Model(name=name, project_id=project_id)
    db.session.add(model)
    db.session.commit()

    return jsonify(model.to_dict()), 201

@model_routes.route('/<int:project_id>/train', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'parameters': [
        {
            'in': 'path',
            'name': 'project_id',
            'type': 'integer',
            'required': True,
            'description': 'Project ID'
        },
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                'properties': {
                    'num_epochs': {
                        'type': 'integer',
                        'description': 'Number of training epochs'
                    },
                    'train_test_split': {
                        'type': 'number',
                        'description': 'Train-test split ratio'
                    },
                    'batch_size': {
                        'type': 'integer',
                        'description': 'Batch size for training'
                    },
                    'model_name': {
                        'type': 'string',
                        'description': 'Model Architecture e.g. ResNet-18'
                    }
                }
            }
        }
    ]
})
def new_training_job(project_id):
    print('running training...')

    project = Project.query.get_or_404(project_id, description="Project ID not found")

    model_name = request.json.get('model_name')
    model_description = request.json.get('model_description')
    num_epochs = int(request.json.get('num_epochs'))
    train_test_split = float(request.json.get('train_test_split'))
    batch_size = int(request.json.get('batch_size'))

    if not (model_name or num_epochs or train_test_split or batch_size):
        return jsonify({'Message': 'Missing required fields'}), 404

    # check if model with the same architecture already exists
    model = Model.query.filter_by(name=model_name, project_id=project.id).first()
    if not model:
        model = Model(name=model_name, project_id=project_id, description=model_description)
        db.session.add(model)
        db.session.commit()

    dataset = Dataset.query.filter_by(project_id=project.id).first()

    print(project)
    print(dataset)
    print(model)

    from app import app
    app_context = app.app_context()

    training_thread = threading.Thread(target=run_training, args=(app_context, project, dataset, model, num_epochs, train_test_split, batch_size))
    training_thread.start()

    return jsonify({'message': 'Training started'}), 200


@model_routes.route('/<int:id>/label', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'parameters': [
        {
            'name': 'id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the model'
        }
    ],
})
def run_model(id):
    print('running model...')

    model = Model.query.get_or_404(id, description="Model ID not found")
    dataset_details = Dataset.query.get_or_404(model.project_id, description="Dataset not found")
    project = Project.query.get_or_404(model.project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    # check that model has been trained
    if not model.saved:
        return jsonify({'Saved model does not exist'}), 404

    from app import app
    app_context = app.app_context()

    training_thread = threading.Thread(target=run_labelling_with_model, args=(app_context, dataset_details, dataset, project))
    training_thread.start()

    return jsonify({'message': 'Job started'}), 200


# for debugging only
@model_routes.route('/all', methods=['GET'])
def get_all_models():
    models = Model.query.all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200


@model_routes.route('<int:model_id>/download', methods=['GET'])
def download_model(model_id):
    model_db = Model.query.get_or_404(model_id, description="Model ID not found")
    project_db = Project.query.get_or_404(model_db.project_id, description="Project ID not found")

    if not model_db.saved:
        return jsonify({'Message': 'Model file not found'}), 404 

    try:
        with TemporaryDirectory() as temp_dir:

            file_name = os.path.basename(model_db.saved)
            print(file_name)
            local_file_path = os.path.join(temp_dir, file_name)

            print(model_db.saved)
            print(local_file_path)

            s3.download_file(project_db.bucket, model_db.saved, local_file_path)
            if os.path.exists(local_file_path):
                return send_file(local_file_path, as_attachment=True)
            else:
                return jsonify({"error": "Downloaded file not found"}), 404
        
    except ClientError as e:
        return jsonify({"error": f"Error downloading file from S3: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        