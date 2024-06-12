from flask import Flask, request, jsonify
from config import Config
from models import db, User, Project

app = Flask(__name__)
app.debug = True

app.config.from_object(Config)
db.init_app(app)

@app.before_request
def create_tables():
    # The following line will remove this handler, making it
    # only run on the first request
    app.before_request_funcs[None].remove(create_tables)

    db.create_all()

@app.route('/')
def hello_world():
   return jsonify({"message": "Welcome to the API!"})

@app.route('/register', methods=['POST'])
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

@app.route('/projects', methods=['POST'])
def create_project():
    data = request.get_json()
    if not data or not 'name' in data or not 'user_id' in data:
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    user = User.query.get(data['user_id'])
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    project = Project(name=data['name'], description=data.get('description', ''), user_id=user.id)
    db.session.add(project)
    db.session.commit()

    return jsonify({"message": "Project created successfully", "project": {"name": project.name, "description": project.description, "user_id": project.user_id}}), 201

@app.route('/users/<int:user_id>/projects', methods=['GET'])
def get_user_projects(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    projects = Project.query.filter_by(user_id=user_id).all()
    projects_list = [{"id": project.id, "name": project.name, "description": project.description} for project in projects]

    return jsonify({"projects": projects_list}), 200

if __name__ == '__main__':
    app.run(debug=True)