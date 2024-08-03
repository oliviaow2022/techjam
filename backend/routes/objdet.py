import os, time
import uuid
import zipfile
import torch
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify
from models import db, Annotation, Project, Dataset, Model, History
from services.S3ImageDataset import s3, ObjDetDataset
from services.objdet import run_training, get_model_instance, split_train_data, get_data_loader
import threading

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
            
            # Upload to S3
            s3.upload_file(local_filepath, os.getenv('S3_BUCKET'), s3_filepath)
            
            # Save to database
            annotation = Annotation(filename=file_name, dataset_id=dataset.id, image_id=i)
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
def train_model(project_id):
    # dataloader = get_data_loader(dataset)
    # train_loader, val_loader = split_train_data(dataloader)

    # model = get_model_instance(num_classes=dataset.num_classes)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # pass
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
    model = Model.query.filter_by(name=model_name, project_id=project.id).first()
    if not model:
        model = Model(name=model_name, project_id=project_id, description=model_description)
        db.session.add(model)
        db.session.commit()
        
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    annotations = Annotation.query.filter_by(dataset_id=dataset.id).with_entities(Annotation.id).all()
    annotation_ids_list = [id[0] for id in annotations]
    dataset = ObjDetDataset(annotation_ids_list, project.bucket, project.prefix)
    dataloader = get_data_loader(dataset)
    train_loader, val_loader = split_train_data(dataloader, batch_size, train_test_split)

    history = History(model_id=model.id)
    db.session.add(history)
    db.session.commit()

    from app import app
    app_context = app.app_context()

    training_thread = threading.Thread(target=run_training, args=(train_loader, val_loader, app_context, project, dataset, model, num_epochs, train_test_split, batch_size))
    training_thread.start()

    return jsonify({'message': 'Training started'}), 200

    # model.to(device)

    # since = time.time()
    # # input from user
    # num_epochs = 10 

    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0

    #     for images, targets in train_loader:
    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         optimizer.zero_grad()
    #         loss_dict = model(images, targets)
    #         losses = sum(loss for loss in loss_dict.values())
    #         losses.backward()
    #         optimizer.step()

    #         running_loss += losses.item()

    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    #     # Validation loop
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = list(image.to(device) for image in images)
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #             loss_dict = model(images, targets)
    #             losses = sum(loss for loss in loss_dict.values())
    #             val_loss += losses.item()

    #     print(f"Validation Loss: {val_loss / len(val_loader)}")

    # # Save the trained model
    # model_save_path = os.path.join('./models', f'{project_id}_model.pth')
    # model_saved = torch.save(model.state_dict(), model_save_path)

    # time_elapsed = time.time() - since
    # print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # s3.upload_file(project.bucket, model_save_path)
    
    # db.session.add(model_saved)
    # db.session.commit()
    # print('model saved to', model_save_path)

    return jsonify({"message": "Model trained successfully", "model_path": model_save_path}), 200

