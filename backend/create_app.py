import os

from flask_socketio import SocketIO
from flask import Flask
from flask_cors import CORS
from config import Config
from flasgger import Swagger
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
from routes.objdet import objdet_routes
from dotenv import load_dotenv
from celery import Celery, Task
from flask import Flask
from models import db

load_dotenv()

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask, broker=os.getenv('REDIS_ENDPOINT'))
    celery_app.config_from_object(app.config["CELERY"])
    print(app.config['CELERY'])
    
    app.extensions["celery"] = celery_app
    return celery_app


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    app.config.from_object(Config)
    app.config.from_prefixed_env()

    # Initialise extensions
    db.init_app(app)
    jwt = JWTManager(app)
    celery_app = celery_init_app(app)
    celery_app.set_default()
    socketio = SocketIO(app, cors_allowed_origins="*")

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

    # Register blueprints
    app.register_blueprint(user_routes, url_prefix='/user')
    app.register_blueprint(project_routes, url_prefix='/project')
    app.register_blueprint(dataset_routes, url_prefix='/dataset')
    app.register_blueprint(data_instance_routes, url_prefix='/instance')
    app.register_blueprint(model_routes, url_prefix='/model')
    app.register_blueprint(history_routes, url_prefix='/history')
    app.register_blueprint(epoch_routes, url_prefix='/epoch')
    app.register_blueprint(senti_routes, url_prefix='/senti')
    app.register_blueprint(objdet_routes, url_prefix='/objdet')
    app.register_blueprint(general_routes)

    return app, celery_app, socketio
