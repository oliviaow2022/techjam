from flask import Blueprint, request, jsonify
from models import db, User, Project, Dataset, Model, DataInstance
from flasgger import swag_from
from flask_jwt_extended import jwt_required

project_routes = Blueprint('project', __name__)

@project_routes.route('/create', methods=['POST'])
@jwt_required()
@swag_from({
    'tags': ['Project'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'The name of the project',
                        'example': 'My Project'
                    },
                    'user_id': {
                        'type': 'integer',
                        'description': 'The ID of the user creating the project',
                        'example': 1
                    },
                    'bucket': {
                        'type': 'string',
                        'description': 'The S3 bucket name (optional)',
                        'example': 'my-bucket'
                    },
                    'prefix': {
                        'type': 'string',
                        'description': 'The S3 bucket prefix (optional)',
                        'example': 'my-prefix'
                    }
                },
                'required': ['name', 'user_id']
            }
        }
    ]}
)
def create_project():
    name = request.json.get('name')
    user_id = request.json.get('user_id')
    bucket = request.json.get('bucket')
    prefix = request.json.get('prefix')
    type = request.json.get('type')

    if not (name or user_id or type):
        return jsonify({"error": "Bad Request", "message": "Missing required fields"}), 400

    user = User.query.get_or_404(user_id, description="User ID not found")

    project = Project(name=name, user_id=user.id, type=type)

    if bucket:
        project.bucket = bucket
    if prefix:
        project.prefix = prefix

    db.session.add(project)
    db.session.commit()

    return jsonify(project.to_dict()), 201


# for debugging only
@project_routes.route('/all', methods=['GET'])
@jwt_required()
def get_all_projects():
    projects = Project.query.all()
    project_list = [project.to_dict() for project in projects]
    return jsonify(project_list), 200


@project_routes.route('/<int:project_id>/models', methods=['GET'])
@jwt_required()
def get_models_for_project(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    models = Model.query.filter(
        Model.project_id == project.id,
        Model.saved.isnot(None)
    ).all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200


@project_routes.route('/<int:project_id>/delete', methods=['DELETE'])
@jwt_required()
def delete_project(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")

    # Fetch and delete all associated datasets
    datasets = Dataset.query.filter_by(project_id=project.id).all()
    for dataset in datasets:
         # Fetch and delete all associated datasets
        data_instances = DataInstance.query.filter_by(dataset_id=dataset.id).all()
        
        for data_instance in data_instances:
            db.session.delete(data_instance)
            db.session.commit()

        db.session.delete(dataset)
        db.session.commit()

    # Fetch and delete all associated models
    models = Model.query.filter_by(project_id=project.id).all()
    for model in models:
        db.session.delete(model)
    
    # Finally, delete the project
    db.session.delete(project)
    db.session.commit()

    return jsonify({'message': 'Project deleted successfully'}), 200

