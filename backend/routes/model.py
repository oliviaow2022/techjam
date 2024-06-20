from flask import Blueprint, request, jsonify
from models import db, Model, Project, Dataset, History, Epoch, DataInstance
from S3ImageDataset import S3ImageDataset
from services.dataset import get_dataframe
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models, datasets
from torch.optim import lr_scheduler
import torch
from services.model import train_model, compute_metrics
from flasgger import swag_from
from io import BytesIO
from S3ImageDataset import download_weights_from_s3

model_routes = Blueprint('model', __name__)

data_transforms = {
    'image_train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'image_val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'fashion_mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,),)
    ]),
    'cifar-10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

@model_routes.route('/create', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'description': 'model name must be in this list [resnet18, densenet121, alexnet, convnext_base]!!',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'project_id': {'type': 'integer'}
                },
                'required': ['name', 'project_id']
            }
        }
    ]
})
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
    'tags': ['Model'],
    'parameters': [
        {
            'in': 'path',
            'name': 'model_id',
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
    print('running training...')

    NUM_EPOCHS = request.json.get('num_epochs')
    TRAIN_TEST_SPLIT = request.json.get('train_test_split')
    BATCH_SIZE = request.json.get('batch_size')

    model = Model.query.get_or_404(id, description="Model ID not found")
    project = Project.query.get_or_404(model.project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    if dataset.name == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=data_transforms['fashion_mnist'])
        val_dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=data_transforms['fashion_mnist'])     
    elif dataset.name == 'cifar-10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['cifar-10'])
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['cifar-10'])
    else:
        # only get datainstances with labels
        df = get_dataframe(dataset.id, return_labelled=True)

        s3_dataset = S3ImageDataset(df, project.bucket, project.prefix, has_labels=True)

        train_size = int(TRAIN_TEST_SPLIT * len(s3_dataset)) 
        val_size = len(s3_dataset) - train_size 

        train_dataset, val_dataset = random_split(s3_dataset, [train_size, val_size])
        train_dataset.dataset.transform = data_transforms['image_train']
        val_dataset.dataset.transform = data_transforms['image_val']

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if model.name == 'resnet18':
        ml_model = models.resnet18(weights='DEFAULT')
        num_ftrs = ml_model.fc.in_features
        ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'densenet121':
        ml_model = models.densenet121(weights='DEFAULT')
        num_ftrs = ml_model.classifier.in_features
        ml_model.classifier = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'alexnet':
        ml_model = models.alexnet(weights='DEFAULT')
        num_ftrs = ml_model.classifier[6].in_features
        ml_model.classifier[6] = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'convnext_base':
        ml_model = models.convnext_base(weights='DEFAULT')
        num_ftrs = ml_model.classifier[2].in_features
        ml_model.classifier[2] = torch.nn.Linear(num_ftrs, dataset.num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ml_model, model_history = train_model(ml_model, model, project, train_dataloader, val_dataloader, criterion, optimizer, exp_lr_scheduler, device, NUM_EPOCHS)

    accuracy, precision, recall, f1 = compute_metrics(ml_model, val_dataloader, device)

    history = History(accuracy=accuracy, precision=precision, recall=recall, f1=f1, model_id=model.id)

    db.session.add(history)
    db.session.commit()

    for i in range(len(model_history)):
        epoch = Epoch(epoch=i, train_acc=model_history[i][0], val_acc=model_history[i][1], train_loss=model_history[i][2], val_loss=model_history[i][3], model_id=model.id, history_id=history.id)
        db.session.add(epoch)
    db.session.commit()

    return jsonify(history.to_dict()), 200


@model_routes.route('/<int:id>/label', methods=['POST'])
@swag_from({
    'tags': ['Model'],
    'parameters': [
        {
            'name': 'id',
            'in': 'path',
            'type': 'integer',
            'required': True,
            'description': 'The ID of the model'
        }
    ],
})
def run_model(id):
    print('running model...')

    model = Model.query.get_or_404(id, description="Model ID not found")
    project = Project.query.get_or_404(model.project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    # initialise model
    if model.name == 'resnet18':
        ml_model = models.resnet18(weights='DEFAULT')
        num_ftrs = ml_model.fc.in_features
        ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)

    if not model.saved:
        return jsonify({'Saved model does not exist'}), 404

    # load in weights
    model_weights = download_weights_from_s3(project.bucket, model.saved)
    ml_model.load_state_dict(torch.load(BytesIO(model_weights)))
    ml_model.eval()

    # only get datainstances with no labels
    df = get_dataframe(dataset.id, return_labelled=False)

    s3_dataset = S3ImageDataset(df, project.bucket, project.prefix, has_labels=False)
    s3_dataset.transform = data_transforms['val']

    dataloader = DataLoader(s3_dataset, batch_size=32, shuffle=True)

    instances_updated = 0

    with torch.no_grad():
        for images, data_instance_ids in dataloader:
            outputs = ml_model(images)
            _, predicted = torch.max(outputs, 1)
            confidence_levels = torch.nn.functional.softmax(outputs, dim=1)
            entropy = -torch.sum(confidence_levels * torch.log2(confidence_levels), dim=1)

            for index, instance_id in enumerate(data_instance_ids):
                data_instance = DataInstance.query.get_or_404(instance_id.item(), description='DataInstance ID not found')
                data_instance.entropy = entropy[index].item()
                data_instance.labels = predicted[index].item()
                db.session.commit()
                instances_updated += 1

    return jsonify({'instances_updated': f'{instances_updated} instances updated'}), 200


# for debugging only
@model_routes.route('/all', methods=['GET'])
def get_all_models():
    models = Model.query.all()
    model_list = [model.to_dict() for model in models]
    return jsonify(model_list), 200