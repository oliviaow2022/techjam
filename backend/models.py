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
    num_classes = db.Column(db.Integer, nullable=False)
    class_to_label_mapping = db.Column(db.JSON, nullable=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    data_instances = db.relationship('DataInstance', backref='dataset', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "num_classes": self.num_classes,
            "class_to_label_mapping": self.class_to_label_mapping
        }

    def __repr__(self):
        return f'<Dataset {self.name}>'


class DataInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(256), nullable=False)
    labels = db.Column(db.String(256), nullable=True) # list of labels separated by commas
    manually_processed = db.Column(db.Boolean, default=False, nullable=False)
    entropy = db.Column(db.Integer, default=0, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "data": self.data,
            "labels": [int(num) for num in self.labels.split(',')] if self.labels and ',' in self.labels else int(self.labels) if self.labels else None,
            "manually_processed": self.manually_processed,
            "entropy": self.entropy,
            "dataset_id": self.dataset_id,
        }

    def __repr__(self):
        return f'<DataInstance {self.data}>'
    
    
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    saved = db.Column(db.String(128), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id
        }

    def __repr__(self):
        return f'<Model {self.name}>'
    

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1 = db.Column(db.Float, nullable=False)
    auc = db.Column(db.Float, nullable=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
        }
    
    
class Epoch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    epoch = db.Column(db.Integer, nullable=False)
    train_acc = db.Column(db.Float, nullable=False)
    val_acc = db.Column(db.Float, nullable=False)
    train_loss = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)
    history_id = db.Column(db.Integer, db.ForeignKey('history.id'), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "epoch": self.epoch,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "model_id": self.model_id,
            "history_id": self.history_id
        }