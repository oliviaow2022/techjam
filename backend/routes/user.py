from flask import Blueprint, request, jsonify
from models import db, User, Project
from flasgger import swag_from
from flask_jwt_extended import create_access_token

user_routes = Blueprint('user', __name__)

@user_routes.route('/register', methods=['POST'])
@swag_from({
    'tags': ['User'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'username': {
                        'type': 'string',
                        'description': 'The username of the user',
                        'example': 'testuser'
                    },
                    'email': {
                        'type': 'string',
                        'description': 'The email of the user',
                        'example': 'test@gmail.com'
                    },
                    'password': {
                        'type': 'string',
                        'description': 'The password of the user',
                        'example': 'testuser'
                    }
                },
                'required': ['username', 'email', 'password']
            }
        }
    ]}
)
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

    return jsonify(user.to_dict()), 201


@user_routes.route('/login', methods=['POST'])
@swag_from({
    'tags': ['User'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'username': {
                        'type': 'string',
                        'example': 'testuser'
                    },
                    'password': {
                        'type': 'string',
                        'example': 'testuser'
                    }
                },
                'required': ['username', 'password']
            }
        }
    ]}
)
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if not (username or password):
        return jsonify({"error": "Bad Request", "message": "Username and password are required"}), 400 

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        access_token = create_access_token(identity=user.id)
        return jsonify({'token': access_token}), 200
    else:
        # Invalid credentials
        return jsonify({'message': 'Login failed. Please check your username and password'}), 401


# for debugging only
@user_routes.route('/all', methods=['GET'])
def get_all_users():
    users = User.query.all()
    user_list = [user.to_dict() for user in users]
    return jsonify(user_list), 200


@user_routes.route('/<int:user_id>/projects', methods=['GET'])
@swag_from({
    'summary': 'Get all projects for a user_id',
    'tags': ['Project'],
    'parameters': [
        {
            'name': 'user_id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'ID of the user'
        }
    ]}
)
def get_user_projects(user_id):
    user = User.query.get_or_404(user_id, description='User ID not found')
    projects = Project.query.filter_by(user_id=user.id).all()
    projects_list = [project.to_dict() for project in projects]
    return jsonify(projects_list), 200

