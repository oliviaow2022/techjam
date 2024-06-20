from flask import Blueprint, request, jsonify
from models import db, User, Project
from flasgger import swag_from

project_routes = Blueprint('project', __name__)

@project_routes.route('/create', methods=['POST'])
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
def get_all_projects():
    projects = Project.query.all()
    project_list = [project.to_dict() for project in projects]
    return jsonify(project_list), 200


@project_routes.route('/<int:project_id>/delete', methods=['GET'])
def delete_project(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    db.session.delete(project)
    db.session.commit()
    return jsonify({'message': 'Project deleted successfully'}), 200

