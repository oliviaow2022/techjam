from flask import Blueprint, request, jsonify, session
from models import db, User, Project
# import jwt
from datetime import datetime
from app import app

user_routes = Blueprint('user', __name__)

@user_routes.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')

    if not (username or email or password):
        return jsonify({"error": "Bad Request", "message": "Username, email, and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Conflict", "message": "Username already exists"}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Conflict", "message": "Email already exists"}), 409

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully", "user": {"username": user.username, "email": user.email}}), 201


@user_routes.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if not (username or password):
        return jsonify({"error": "Bad Request", "message": "Username and password are required"}), 400 

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session['user_id'] = user.id  # Store user ID in session
        # token = jwt.encode({'user_id': user.id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, app.config['SECRET_KEY'])
        # return jsonify({'token': token}), 200
    else:
        # Invalid credentials
        return jsonify({'message': 'Login failed. Please check your username and password'}), 401


@user_routes.route('/all', methods=['GET'])
def get_all_users():
    users = User.query.all()

    user_list = []
    for user in users:
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
        user_list.append(user_data)

    return jsonify({'users': user_list}), 200



@user_routes.route('/<int:user_id>/projects', methods=['GET'])
def get_user_projects(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    projects = Project.query.filter_by(user_id=user_id).all()
    projects_list = [{"id": project.id, "name": project.name} for project in projects]

    return jsonify({"projects": projects_list}), 200

