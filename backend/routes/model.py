from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset
from S3ImageDataset import S3ImageDataset
from services.dataset import get_dataframe
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models
from torch.optim import lr_scheduler
import torch
from services.model import train_model
from flasgger import swag_from

model_routes = Blueprint('model', __name__)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

@model_routes.route('/create', methods=['POST'])
def create_model():
    name = request.json.get('name')
    project_id = request.json.get('project_id')

    if not (name or project_id):
        return jsonify({"error": "Bad Request", "message": "Name and user_id are required"}), 400

    model = Model(name=name, project_id=project_id)
    db.session.add(model)
    db.session.commit()

    return jsonify(model.to_dict()), 201


@model_routes.route('/<int:id>/train', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'in': 'path',
            'name': 'id',
            'type': 'integer',
            'required': True,
            'description': 'ID of the model to train'
        },
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                'properties': {
                    'num_epochs': {
                        'type': 'integer',
                        'description': 'Number of training epochs'
                    },
                    'train_test_split': {
                        'type': 'number',
                        'description': 'Train-test split ratio'
                    },
                    'batch_size': {
                        'type': 'integer',
                        'description': 'Batch size for training'
                    }
                }
            }
        }
    ]
})
def run_training(id):
    NUM_EPOCHS = request.json.get('num_epochs')
    TRAIN_TEST_SPLIT = request.json.get('train_test_split')
    BATCH_SIZE = request.json.get('batch_size')

    model = Model.query.get_or_404(id, description="Model ID not found")
    project = Project.query.get_or_404(model.project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    # only get datainstances with labels
    df = get_dataframe(dataset.id, return_all=False)

    s3_dataset = S3ImageDataset(df, project.bucket, project.prefix)

    train_size = int(TRAIN_TEST_SPLIT * len(s3_dataset)) 
    val_size = len(s3_dataset) - train_size 

    train_dataset, val_dataset = random_split(s3_dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ml_model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = ml_model.fc.in_features
    ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ml_model, history = train_model(ml_model, train_dataloader, val_dataloader, criterion, optimizer, exp_lr_scheduler, device, NUM_EPOCHS)

    return 200


@model_routes.route('/<int:id>/label', methods=['GET'])
def run_model():
    return


# for debugging only
@model_routes.route('/all', methods=['GET'])
def get_all_models():
    models = Model.query.all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200