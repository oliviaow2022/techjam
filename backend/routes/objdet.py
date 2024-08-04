import os
import zipfile
import torch
import json

from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify
from models import db, Annotation, Project, Dataset, Model, History
from services.S3ImageDataset import s3, ObjDetDataset
from services.objdet import run_training

objdet_routes = Blueprint('objdet', __name__)

@objdet_routes.route('<int:dataset_id>/upload', methods=['POST'])
def upload_file(dataset_id):
    print('UPLOADING FILE')
    print(request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")
    print(dataset)
    project = Project.query.get_or_404(dataset.project_id, description="Project ID not found")
    print(project)

    file = request.files['file']
    print(file.filename)

    UPLOAD_FOLDER = './tmp'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Sanitize filename
    original_filename = secure_filename(file.filename)
    local_zip_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(local_zip_path)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_FOLDER)

    os.remove(local_zip_path)  # Remove the zip file after extraction

    # Process each file in the extracted folder
    for root, _, files in os.walk(UPLOAD_FOLDER):
        for i, file_name in enumerate(files):
            if file_name.endswith('.zip'):  # Skip the original zip file
                continue

            local_filepath = os.path.join(root, file_name)
            # s3_filepath = os.path.join(project.prefix, unique_filename)
            s3_filepath = f'{project.prefix}/{file_name}'
            print(s3_filepath)
            
            # Upload to S3
            s3.upload_file(local_filepath, os.getenv('S3_BUCKET'), s3_filepath)
            
            # Save to database
            annotation = Annotation(filename=file_name, dataset_id=dataset.id, image_id=i)
            print(annotation.to_dict())
            db.session.add(annotation)
            db.session.commit()

            # Remove file from local storage
            os.remove(local_filepath)

    return jsonify({'message': 'Files successfully uploaded into dataset'}), 201


# for debugging only
@objdet_routes.route('<int:dataset_id>/all', methods=['GET'])
def view_annotations(dataset_id):
    annotations = Annotation.query.filter_by(dataset_id=dataset_id)
    return [a.to_dict() for a in annotations]

    
@objdet_routes.route('<int:project_id>/batch', methods=['POST'])
def return_batch(project_id): 
    batch_size = request.json.get('batch_size')

    if not batch_size:
        batch_size = 20
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    annotations = Annotation.query.filter(
        Annotation.dataset_id == dataset.id,
        Annotation.manually_processed == False
    ).order_by(Annotation.confidence.asc()).limit(batch_size).all()

    if len(annotations) < batch_size:
        top_up_count = batch_size - len(annotations)
        top_up_annotations = Annotation.query.filter(
            Annotation.dataset_id == dataset.id,
            Annotation.manually_processed == True
        ).order_by(Annotation.confidence.asc()).limit(top_up_count).all()

        annotations.extend(top_up_annotations)
        
    data_list = [instance.process_bbox() for instance in annotations]

    return jsonify(data_list), 200


@objdet_routes.route('<int:annotation_id>/label', methods=['POST'])
def label_data(annotation_id):
    # format from frontend: [{x1: 74, y1: 62, x2: 401, y2: 365, label: 'bus'}]
    data = request.json.get('annotations')
    image_display_ratio = request.json.get('image_display_ratio')

    if not image_display_ratio:
        image_display_ratio = 1

    annotation = Annotation.query.get_or_404(annotation_id)

    parsed_boxes = []
    parsed_labels = []
    parsed_area = []
    parsed_iscrowd = []

    for item in data:
        # scale coordinates from canvas size to original image size
        xmin = round(item['x1'] * image_display_ratio)
        ymin = round(item['y1'] * image_display_ratio)
        xmax = round(item['x2'] * image_display_ratio)
        ymax = round(item['y2'] * image_display_ratio)
        label = item['label']

        parsed_boxes.append([xmin, ymin, xmax, ymax])
        parsed_labels.append(label)
        parsed_area.append((xmax - xmin) * (ymax - ymin))
        parsed_iscrowd.append(0)    # Assuming iscrowd is 0

    annotation.boxes = parsed_boxes
    annotation.labels = parsed_labels
    annotation.area = parsed_area
    annotation.iscrowd = parsed_iscrowd
    annotation.manually_processed = True

    db.session.commit()

    return jsonify({"message": "Annotation updated successfully", "annotation": annotation.to_dict()}), 200


@objdet_routes.route('<int:project_id>/train', methods=['POST'])
def run_training_model(project_id):

    print('running training...')

    project = Project.query.get_or_404(project_id, description="Project ID not found")

    model_name = request.json.get('model_name')
    model_description = request.json.get('model_description')
    num_epochs = int(request.json.get('num_epochs'))
    train_test_split = float(request.json.get('train_test_split'))
    batch_size = int(request.json.get('batch_size'))

    if not (model_name or num_epochs or train_test_split or batch_size):
        return jsonify({'Message': 'Missing required fields'}), 404

    # check if model with the same architecture already exists
    model_db = Model.query.filter_by(name=model_name, project_id=project.id).first()
    if not model_db:
        model_db = Model(name=model_name, project_id=project_id, description=model_description)
        db.session.add(model_db)
        db.session.commit()
        
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    annotations = Annotation.query.filter_by(dataset_id=dataset.id).filter(Annotation.labels.isnot(None)).with_entities(Annotation.id).all()
    annotation_ids = [annotation.id for annotation in annotations]

    if len(annotation_ids) == 0:
        return jsonify({'Message': 'No labelled data to train with'}), 404

    print(dataset.to_dict())
    history = History(model_id=model_db.id)
    db.session.add(history)
    db.session.commit()

    task = run_training.delay(annotation_ids, project.to_dict(), dataset.to_dict(), model_db.to_dict(), history.to_dict(), num_epochs, batch_size, train_test_split)

    history.task_id = task.id
    db.session.commit()

    return jsonify({'task_id': task.id}), 200

