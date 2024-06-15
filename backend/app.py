from flask import Flask, jsonify
from config import Config
import csv
from models import db, User, Project, Dataset, DataInstance, Model
from routes.user import user_routes
from routes.project import project_routes
from routes.dataset import dataset_routes
from routes.data_instance import data_instance_routes
from routes.model import model_routes
import click
from flasgger import Swagger

app = Flask(__name__)
app.register_blueprint(user_routes, url_prefix='/user')
app.register_blueprint(project_routes, url_prefix='/project')
app.register_blueprint(dataset_routes, url_prefix='/dataset')
app.register_blueprint(data_instance_routes, url_prefix='/instance')
app.register_blueprint(model_routes, url_prefix='/model')

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

@app.before_request
def create_tables():
    # The following line will remove this handler, making it
    # only run on the first request
    app.before_request_funcs[None].remove(create_tables)

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

    dataset = Dataset(name="Ants and Bees", project_id=project.id, num_classes=2)

    db.session.add(dataset)
    db.session.commit()

    model = Model(name='resnet18', project_id=project.id)

    db.session.add(model)
    db.session.commit()

    """Seed the database from a CSV file."""
    file_path = 'dataset.csv'
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_instance = DataInstance(
                data=row['filename'],
                labels=row['label'],
                dataset_id=1
            )
            db.session.add(data_instance)
        db.session.commit()

    click.echo('Seed data added successfully.')


@app.route('/')
def hello_world():
   return jsonify({"message": "Welcome to the API!"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)