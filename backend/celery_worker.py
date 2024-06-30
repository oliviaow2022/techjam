from celery import Celery, Task
from flask import Flask
import os

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask, broker=os.getenv('REDIS_ENDPOINT'))
    celery_app.config_from_object(app.config["CELERY"])
    print(app.config['CELERY'])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app