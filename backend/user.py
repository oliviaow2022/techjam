from flask import Blueprint, request, jsonify
from models import db, User

user_routes = Blueprint('user', __name__)

@user_routes.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not 'username' in data or not 'email' in data or not 'password' in data:
        return jsonify({"error": "Bad Request", "message": "Username, email, and password are required"}), 400

    username = data['username']
    email = data['email']
    password = data['password']

    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({"error": "Conflict", "message": "Username or email already exists"}), 409

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully", "user": {"username": user.username, "email": user.email}}), 201


@user_routes.route('/login', methods=['POST'])
def login():
    # Login logic
    pass


@user_routes.route('/<int:user_id>/projects', methods=['GET'])
def get_user_projects(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    projects = Project.query.filter_by(user_id=user_id).all()
    projects_list = [{"id": project.id, "name": project.name, "description": project.description} for project in projects]

    return jsonify({"projects": projects_list}), 200

