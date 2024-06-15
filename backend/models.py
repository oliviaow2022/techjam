from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username
        }

    def __repr__(self):
        return f'<User {self.username}>'
    
    
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    bucket = db.Column(db.String(128), nullable=True)
    prefix = db.Column(db.String(128), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    datasets = db.relationship('Dataset', backref='project', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id
        }

    def __repr__(self):
        return f'<Project {self.name}>'
    

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    data_instances = db.relationship('DataInstance', backref='dataset', lazy=True)
    num_classes = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "num_classes": self.num_classes
        }

    def __repr__(self):
        return f'<Dataset {self.name}>'


class DataInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(256), nullable=False)
    labels = db.Column(db.String(256), nullable=True) # list of labels separated by commas
    manually_processed = db.Column(db.Boolean, default=False, nullable=False)
    confidence = db.Column(db.Integer, default=0, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "data": self.data,
            "labels": [int(num) for num in self.labels.split(',')] if ',' in self.labels else int(self.labels) if self.labels else None,
            "manually_processed": self.manually_processed,
            "confidence": self.confidence,
            "dataset_id": self.dataset_id,
        }

    def __repr__(self):
        return f'<DataInstance {self.id}>'
    
    
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id
        }

    def __repr__(self):
        return f'<Model {self.id}>'