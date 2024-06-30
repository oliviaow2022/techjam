from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset
from services.model import run_training, run_labelling_using_model, add_together
from flasgger import swag_from
import threading

model_routes = Blueprint('model', __name__)

@model_routes.route('/create', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'description': 'model name must be in this list [resnet18, densenet121, alexnet, convnext_base]!!',
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
                        'description': 'Model Architecture e.g. resnet18'
                    }
                }
            }
        }
    ]
})
def new_training_job(project_id):
    print('running training...')

    model_name = request.json.get('model_name')
    num_epochs = request.json.get('num_epochs')
    train_test_split = request.json.get('train_test_split')
    batch_size = request.json.get('batch_size')

    if not (model_name or num_epochs or train_test_split or batch_size):
        return jsonify({'Message': 'Missing required fields'}), 404

    result = run_training.delay(project_id, model_name, num_epochs, train_test_split, batch_size)

    return jsonify({'message': 'Training started', 'job_id': result.id}), 200



@model_routes.route('/test_celery', methods=['GET'])
def test_celery():
    a = 2
    b = 3
    result = add_together.delay(a, b)
    return {"result_id": result.id}


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
    project = Project.query.get_or_404(model.project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    # check that model has been trained
    if not model.saved:
        return jsonify({'Saved model does not exist'}), 404

    training_thread = threading.Thread(target=run_labelling_using_model, args=(app_context, project, dataset, model))
    training_thread.start()

    return jsonify({'message': 'Job started'}), 200


# for debugging only
@model_routes.route('/all', methods=['GET'])
def get_all_models():
    models = Model.query.all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200