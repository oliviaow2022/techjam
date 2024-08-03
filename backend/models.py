from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import jsonify
import json

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
    type = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    bucket = db.Column(db.String(128), nullable=True)
    prefix = db.Column(db.String(128), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    datasets = db.relationship('Dataset', backref='project', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "bucket": self.bucket,
            "type": self.type,
            "prefix": self.prefix,
            "bucket": self.bucket
        }

    def __repr__(self):
        return f'<Project {self.name}>'
    

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=True)
    num_classes = db.Column(db.Integer, nullable=False)
    class_to_label_mapping = db.Column(db.JSON, nullable=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    data_instances = db.relationship('DataInstance', backref='dataset', lazy=True)

    def to_dict(self):
        project_type = Project.query.filter_by(id=self.project_id).first().type
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "num_classes": self.num_classes,
            "class_to_label_mapping": self.class_to_label_mapping,
            "project_type": project_type
        }

    def __repr__(self):
        return f'<Dataset {self.to_dict()}>'


class DataInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(256), nullable=False)
    labels = db.Column(db.String(256), nullable=True) # list of labels separated by commas
    manually_processed = db.Column(db.Boolean, default=False, nullable=False)
    entropy = db.Column(db.Integer, default=0, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)

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
    

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String, nullable=False)
    image_id = db.Column(db.Integer, nullable=True)
    boxes = db.Column(db.JSON, nullable=True)
    labels = db.Column(db.JSON, nullable=True)
    area = db.Column(db.JSON, nullable=True)
    iscrowd = db.Column(db.JSON, nullable=True)
    manually_processed = db.Column(db.Boolean, default=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    confidence = db.Column(db.Float, default=0.0, nullable=False)

    def process_bbox(self):
        bboxes = []
        if self.boxes and self.labels:
            for box, label in zip(self.boxes, self.labels):
                bbox = {
                    'x1': box[0],
                    'y1': box[1],
                    'x2': box[2],
                    'y2': box[3],
                    'label': label
                }
                bboxes.append(bbox)

        return {
            "id": self.id,
            "filename": self.filename,
            "image_id": self.image_id,
            "bboxes": bboxes,
            "dataset_id": self.dataset_id
        }


    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "image_id": self.image_id,
            "boxes": self.boxes,
            "labels": self.labels,
            "area": self.area,
            "iscrowd": self.iscrowd,
            "dataset_id": self.dataset_id
        }

    def __repr__(self):
        return f'<Annotation id={self.id}, image_id={self.image_id}, boxes={self.boxes}, labels={self.labels}, area={self.area}, iscrowd={self.iscrowd}>'
    
    
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    description = db.Column(db.String(256), nullable=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    saved = db.Column(db.String(128), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "saved": self.saved
        }

    def __repr__(self):
        return f'<Model {self.to_dict()}>'
    

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1 = db.Column(db.Float, nullable=True)
    auc = db.Column(db.Float, nullable=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    task_id = db.Column(db.String(256), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "created_at": self.created_at,
            "task_id": self.task_id
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