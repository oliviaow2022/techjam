from flask import Blueprint, request, jsonify
from models import db, User, Project

project_routes = Blueprint('project', __name__)

@project_routes.route('/create', methods=['POST'])
def create_project():
    name = request.json.get('name')
    user_id = request.json.get('user_id')

    if not (name or user_id):
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    user = User.query.get_or_404(user_id, description="User ID not found")

    project = Project(name=name, user_id=user.id)
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

