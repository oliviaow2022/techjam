import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY=os.urandom(16)
    JWT_TOKEN_LOCATION=['headers']
    CELERY=dict(
        broker_url=os.getenv('REDIS_ENDPOINT'),
        result_backend=os.getenv('REDIS_ENDPOINT'),
        task_ignore_result=False,
    ),