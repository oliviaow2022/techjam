import os
import threading

from flask import Blueprint, request, jsonify, send_file
from models import db, Model, Project, Dataset, History
from services.model import run_training, run_labelling_using_model
from flasgger import swag_from
import threading
from services.S3ImageDataset import s3
from tempfile import TemporaryDirectory
from botocore.exceptions import ClientError
from flask_jwt_extended import jwt_required


model_routes = Blueprint('model', __name__)


@model_routes.route('/<int:project_id>/train', methods=['POST'])
@jwt_required()
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

    history = History(model_id=model.id)
    db.session.add(history)
    db.session.commit()

    task = run_training.delay(project.to_dict(), dataset.to_dict(), model.to_dict(), history.to_dict(), num_epochs, train_test_split, batch_size)

    history.task_id = task.id
    db.session.commit()

    return jsonify({'task_id': task.id}), 200


# for debugging only
@model_routes.route('/all', methods=['GET'])
@jwt_required()
def get_all_models():
    models = Model.query.all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200


@model_routes.route('<int:history_id>/download', methods=['GET'])
@jwt_required()
def download_model(history_id):
    history = History.query.get_or_404(history_id)
    model_db = Model.query.get_or_404(history.model_id)
    project_db = Project.query.get_or_404(model_db.project_id)

    if not history.model_path:
        return jsonify({'Message': 'Model file not found'}), 404 

    try:
        with TemporaryDirectory() as temp_dir:

            file_name = os.path.basename(history.model_path)
            print(file_name)
            local_file_path = os.path.join(temp_dir, file_name)

            print(history.model_path)
            print(local_file_path)

            s3.download_file(project_db.bucket, history.model_path, local_file_path)
            if os.path.exists(local_file_path):
                return send_file(local_file_path, as_attachment=True)
            else:
                return jsonify({"error": "Downloaded file not found"}), 404
        
    except ClientError as e:
        return jsonify({"error": f"Error downloading file from S3: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        