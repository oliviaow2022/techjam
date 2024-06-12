from flask import Blueprint, request, jsonify
from models import db, User, Project

project_routes = Blueprint('project', __name__)

@project_routes.route('/create', methods=['POST'])
def create_project():
    data = request.get_json()
    if not data or not 'name' in data or not 'user_id' in data:
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    user = User.query.get(data['user_id'])
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    project = Project(name=data['name'], user_id=user.id)
    db.session.add(project)
    db.session.commit()

    return jsonify({"message": "Project created successfully", "project": {"name": project.name, "description": project.description, "user_id": project.user_id}}), 201
