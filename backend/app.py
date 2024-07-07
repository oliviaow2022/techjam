from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
import csv
from models import db, User, Project, Dataset, DataInstance, Model
import click
from flasgger import Swagger
from flask.cli import with_appcontext
from flask_jwt_extended import JWTManager
from routes.user import user_routes
from routes.project import project_routes
from routes.dataset import dataset_routes
from routes.data_instance import data_instance_routes
from routes.model import model_routes
from routes.history import history_routes
from routes.epoch import epoch_routes
from routes.general import general_routes
from routes.senti import senti_routes
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)
app.register_blueprint(user_routes, url_prefix='/user')
app.register_blueprint(project_routes, url_prefix='/project')
app.register_blueprint(dataset_routes, url_prefix='/dataset')
app.register_blueprint(data_instance_routes, url_prefix='/instance')
app.register_blueprint(model_routes, url_prefix='/model')
app.register_blueprint(history_routes, url_prefix='/history')
app.register_blueprint(epoch_routes, url_prefix='/epoch')
app.register_blueprint(senti_routes, url_prefix='/senti')
app.register_blueprint(general_routes)

app.config.from_object(Config)
db.init_app(app)
jwt = JWTManager(app)

# Initialize Swagger
swagger_config = {
    'swagger': '2.0',
    'info': {
        'title': 'Labella',
        'description': 'API documentation using Swagger and Flask',
        'version': '1.0'
    },
    'host': '127.0.0.1:5001'
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

    ants_bees = Project(name="Multi-Class Classification", user_id=user.id, bucket=os.getenv('S3_BUCKET'), prefix='transfer-antsbees', type="Single Label Classification")
    fashion_mnist = Project(name="Fashion Mnist", user_id=user.id, bucket=os.getenv('S3_BUCKET'), prefix='fashion-mnist', type="Single Label Classification")
    cifar10 = Project(name="Cifar-10", user_id=user.id, type="1")
    db.session.add_all([ants_bees, fashion_mnist, cifar10])
    db.session.commit()

    ants_bees_ds = Dataset(name="Ants and Bees", project_id=ants_bees.id, num_classes=2, class_to_label_mapping={0: 'ants', 1: 'bees'})
    db.session.add(ants_bees_ds)
    db.session.commit()

    resnet18 = Model(name='ResNet-18', project_id=ants_bees.id)
    densenet121 = Model(name='DenseNet-121', project_id=ants_bees.id)
    alexnet = Model(name='AlexNet', project_id=ants_bees.id)
    convnext_base = Model(name='ConvNext Base', project_id=ants_bees.id)
    resnet18_2 = Model(name='ResNet-18', project_id=fashion_mnist.id)
    resnet18_3 = Model(name='ResNet-18', project_id=cifar10.id)
    db.session.add_all([resnet18, densenet121, alexnet, convnext_base, resnet18_2, resnet18_3])
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


if __name__ == '__main__':
    app.run(debug=True, port=5001)