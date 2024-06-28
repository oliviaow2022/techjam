from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, User
from flasgger import swag_from
from services.dataset import get_dataset_config
import json
from flask_jwt_extended import jwt_required, get_jwt_identity

general_routes = Blueprint('general', __name__)

@general_routes.route('/create', methods=['POST'])
@jwt_required()
@swag_from({
    'summary': 'Create a project and dataset in one API call',
    'security': [{'Bearer': []}],
    'parameters': [
        {
            'name': 'Authorization',
            'in': 'header',
            'type': 'string',
            'required': True,
            'description': 'Bearer token for authentication'
        },
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'project_name': {'type': 'string', 'description': 'Name of the project', 'example': 'Project A'},
                    'project_type': {'type': 'string', 'description': 'Name of the project', 'example': 'Multi-Class Classification'},
                    'user_id': {'type': 'integer', 'description': 'ID of the user', 'example': 1},
                    's3_bucket': {'type': 'string', 'description': 'S3 bucket name', 'example': 'my-s3-bucket'},
                    's3_prefix': {'type': 'string', 'description': 'S3 prefix path', 'example': 'my-prefix/'},
                    'dataset_name': {'type': 'string', 'description': 'Name of the dataset', 'example': 'Dataset A'},
                    'num_classes': {'type': 'integer', 'description': 'Number of classes in the dataset', 'example': 2},
                    'class_to_label_mapping': {
                        'type': 'object',
                        'description': 'Mapping of class indices to labels',
                        'example': {0: 'class_a', 1: 'class_b'}
                    },
                },
                'required': ['project_name', 'user_id', 'dataset_name', 'num_classes', 'class_to_label_mapping']
            }
        }
    ]
})
def create_project_dataset():
    project_name = request.json.get('project_name')
    project_type = request.json.get('project_type')
    user_id = get_jwt_identity()
    s3_bucket = request.json.get('s3_bucket')
    s3_prefix = request.json.get('s3_prefix')
    dataset_name = request.json.get('dataset_name')
    num_classes = request.json.get('num_classes')
    class_to_label_mapping = request.json.get('class_to_label_mapping')

    num_classes, class_to_label_mapping = get_dataset_config(project_name, num_classes, class_to_label_mapping)

    # Validate input
    if not all([project_name, project_type, user_id, dataset_name, num_classes, class_to_label_mapping]):
        return jsonify({"error": "Bad Request", "message": "Missing required fields"}), 400

    user = User.query.get_or_404(user_id, description="User ID not found")
    project = Project(name=project_name, user_id=user.id, bucket=s3_bucket, prefix=s3_prefix, type=project_type)
    db.session.add(project)
    db.session.commit()

    dataset = Dataset(name=dataset_name, project_id=project.id, num_classes=num_classes, class_to_label_mapping=json.dumps(class_to_label_mapping))
    db.session.add(dataset)
    db.session.commit()

    return jsonify({
        'project': project.to_dict(), 
        'dataset': dataset.to_dict()
    }), 201