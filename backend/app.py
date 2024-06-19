from flask import Flask, jsonify, request
from config import Config
import csv
from models import db, User, Project, Dataset, DataInstance, Model
from routes.user import user_routes
from routes.project import project_routes
from routes.dataset import dataset_routes
from routes.data_instance import data_instance_routes
from routes.model import model_routes
from routes.history import history_routes
from routes.epoch import epoch_routes
import click
from flasgger import Swagger, swag_from
from flask.cli import with_appcontext
import json

app = Flask(__name__)
app.register_blueprint(user_routes, url_prefix='/user')
app.register_blueprint(project_routes, url_prefix='/project')
app.register_blueprint(dataset_routes, url_prefix='/dataset')
app.register_blueprint(data_instance_routes, url_prefix='/instance')
app.register_blueprint(model_routes, url_prefix='/model')
app.register_blueprint(history_routes, url_prefix='/history')
app.register_blueprint(epoch_routes, url_prefix='/epoch')

app.config.from_object(Config)
db.init_app(app)

# Initialize Swagger
swagger_config = {
    'swagger': '2.0',
    'info': {
        'title': 'Labella',
        'description': 'API documentation using Swagger and Flask',
        'version': '1.0'
    },
    'host': '127.0.0.1:5001',
    'schemes': ['http', 'https']
}
Swagger(app, template=swagger_config)


@app.cli.command('seed')
@with_appcontext
def seed():
    db.drop_all()
    db.create_all()

    """Seed the database."""
    user = User(email='test@gmail.com', username='testuser')
    user.set_password('testuser')
    db.session.add(user)
    db.session.commit()

    project = Project(name="Multi-Class Classification", user_id=user.id, bucket='dltechjam', prefix='transfer-antsbees')
    db.session.add(project)
    db.session.commit()

    dataset = Dataset(name="Ants and Bees", project_id=project.id, num_classes=2, class_to_label_mapping={0: 'ants', 1: 'bees'})
    db.session.add(dataset)
    db.session.commit()

    resnet18 = Model(name='resnet18', project_id=project.id)
    densenet121 = Model(name='densenet121', project_id=project.id)
    alexnet = Model(name='alexnet', project_id=project.id)
    convnext_base = Model(name='convnext_base', project_id=project.id)
    db.session.add_all([resnet18, densenet121, alexnet, convnext_base])
    db.session.commit()

    """Seed the database from a CSV file."""
    file_path = 'dataset.csv'
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_instance = DataInstance(
                data=row['filename'],
                labels=row['label'],
                dataset_id=1,
                manually_processed=bool(row.get('manually_processed', False))
            )
            db.session.add(data_instance)
        db.session.commit()

    click.echo('Seed data added successfully.')


@app.route('/')
def hello_world():
   return jsonify({"message": "Welcome to the API!"})


@app.route('/create', methods=['POST'])
@swag_from({
    'description': 'Create a project, dataset, and model in one API call.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'project_name': {'type': 'string', 'description': 'Name of the project', 'example': 'Project A'},
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
                    'model_name': {'type': 'string', 'description': 'Name of the model', 'example': 'resnet18'}
                },
                'required': ['project_name', 'user_id', 's3_bucket', 's3_prefix', 'dataset_name', 'num_classes', 'class_to_label_mapping', 'model_name']
            }
        }
    ]
})
def create_project_dataset_model():
    project_name = request.json.get('project_name')
    user_id = request.json.get('user_id')
    s3_bucket = request.json.get('s3_bucket')
    s3_prefix = request.json.get('s3_prefix')
    dataset_name = request.json.get('dataset_name')
    num_classes = request.json.get('num_classes')
    class_to_label_mapping = request.json.get('class_to_label_mapping')
    model_name = request.json.get('model_name')

    # Validate input
    if not all([project_name, user_id, s3_bucket, s3_prefix, dataset_name, num_classes, class_to_label_mapping]):
        return jsonify({"error": "Bad Request", "message": "Missing required fields"}), 400

    user = User.query.get_or_404(user_id, description="User ID not found")
    project = Project(name=project_name, user_id=user.id, bucket=s3_bucket, prefix=s3_prefix)
    db.session.add(project)
    db.session.commit()

    dataset = Dataset(name=dataset_name, project_id=project.id, num_classes=num_classes, class_to_label_mapping=json.dumps(class_to_label_mapping))
    db.session.add(dataset)
    db.session.commit()

    model = Model(name=model_name, project_id=project.id)
    db.session.add(model)
    db.session.commit()

    return jsonify({
        'project': project.to_dict(), 
        'dataset': dataset.to_dict(), 
        'model': model.to_dict()}
    ), 201

if __name__ == '__main__':
    app.run(debug=True, port=5001)