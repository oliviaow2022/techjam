import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY='we2314jon3b2r9uf0qw8'
    JWT_SECRET_KEY='huh'
    JWT_TOKEN_LOCATION=['headers']