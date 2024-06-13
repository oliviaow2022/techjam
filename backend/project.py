from flask import Blueprint, request, jsonify
from models import db, User, Project

project_routes = Blueprint('project', __name__)

@project_routes.route('/create', methods=['POST'])
def create_project():
    name = request.json.get('name')
    user_id = request.json.get('user_id')

    if not (name or user_id):
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    project = Project(name=name, user_id=user.id)
    db.session.add(project)
    db.session.commit()

    return jsonify({"message": "Project created successfully", "project": {"name": project.name, "user_id": project.user_id}}), 201


@project_routes.route('/all', methods=['GET'])
def get_all_projects():
    projects = Project.query.all()

    project_list = []
    for project in projects:
        project_data = {
            'id': project.id,
            'name': project.name,
            'user_id': project.user_id
        }
        project_list.append(project_data)

    return jsonify({'projects': project_list}), 200


@project_routes.route('/<int:project_id>/delete', methods=['GET'])
def delete_project(project_id):
    project = Project.query.get(project_id)

    if not project:
        return jsonify({"error": "Not Found", "message": "Project not found"}), 404
    
    db.session.delete(project)
    db.session.commit()
    return jsonify({'message': 'Project deleted successfully'}), 200

