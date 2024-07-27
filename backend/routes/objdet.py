import os
import uuid
import zipfile
import torch

from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify
from models import db, Annotation, Project, Dataset
from services.S3ImageDataset import s3, ObjDetDataset
from services.objdet import get_model_instance, split_train_data, get_data_loader

objdet_routes = Blueprint('objdet', __name__)

@objdet_routes.route('<int:dataset_id>/upload', methods=['POST'])
def upload_file(dataset_id):
    print(request.files)

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")
    project = Project.query.get_or_404(dataset.project_id, description="Project ID not found")

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
            unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(file_name)[1]}"
            # s3_filepath = os.path.join(project.prefix, unique_filename)
            s3_filepath = f'{project.prefix}/{unique_filename}'
            
            # Upload to S3
            s3.upload_file(local_filepath, os.getenv('S3_BUCKET'), s3_filepath)
            
            # Save to database
            annotation = Annotation(filename=unique_filename, dataset_id=dataset.id, image_id=i)
            db.session.add(annotation)

            # Remove file from local storage
            os.remove(local_filepath)

    db.session.commit()

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
        
    data_list = [instance.to_dict() for instance in annotations]

    return jsonify(data_list), 200


@objdet_routes.route('<int:annotation_id>/label', methods=['POST'])
def label_data(annotation_id):
    # format from frontend: [{x1: 74, y1: 62, x2: 401, y2: 365, label: 'bus'}]
    data = request.json.get('annotations')
    image_display_ratio = request.json.get('image_display_ratio')

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
def train_model(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    annotations = Annotation.query.filter_by(dataset_id=dataset.id).with_entities(Annotation.id).all()
    annotation_ids_list = [id[0] for id in annotations]

    dataset = ObjDetDataset(annotation_ids_list, project.bucket, project.prefix)
    dataloader = get_data_loader(dataset)
    train_loader, val_loader = split_train_data(dataloader)

    model = get_model_instance(num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pass
