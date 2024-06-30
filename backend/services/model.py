import torch
import os, time
from tempfile import TemporaryDirectory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from S3ImageDataset import s3, S3ImageDataset, download_weights_from_s3
from models import db, History, Epoch, DataInstance, Project, Dataset, Model
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torchvision
from torch.optim import lr_scheduler
from services.dataset import get_dataframe
from io import BytesIO
from celery import shared_task

data_transforms = {
    'image_train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'image_val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'fashion_mnist': torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'cifar-10': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


def train_model(model, model_db, project, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=3):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        history = []

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
            scheduler.step()
            val_loss, val_acc = validate_epoch(model, val_dataloader, criterion, device)

            # Cache metrics for later comparison
            history.append([train_acc, val_acc, train_loss, val_loss])
            print(f'Epoch {epoch}/{num_epochs - 1}: '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

            # Deep copy the model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Val Acc: {best_acc:.4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        # upload model to s3
        model_path = f'{project.prefix}/{model_db.name}.pth'
        s3.upload_file(best_model_params_path, project.bucket, model_path)
        
        # save model to db
        model_db.saved = model_path
        db.session.commit()
        print('model saved to', model_path)

    return model, history


def compute_metrics(model, dataloader, device):
    model.eval()

    y_labels = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            y_labels.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_labels, y_pred)
    precision = precision_score(y_labels, y_pred)
    recall = recall_score(y_labels, y_pred)
    f1 = f1_score(y_labels, y_pred)

    return accuracy, precision, recall, f1

@shared_task(ignore_result=False)
def add_together(a: int, b: int) -> int:
    return a + b


@shared_task(ignore_result=False)
def run_training(project_id, model_name, NUM_EPOCHS, TRAIN_TEST_SPLIT, BATCH_SIZE):

    project = Project.query.get_or_404(project_id, description="Project ID not found")

     # check if model with the same architecture already exists
    model = Model.query.filter_by(name=model_name, project_id=project.id).first()
    if not model:
        model = Model(name=model_name, project_id=project_id)
        db.session.add(model)
        db.session.commit()

    dataset = Dataset.query.filter_by(project_id=project.id).first()

    if dataset.name == 'fashion-mnist':
        train_dataset = torchvision.datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=data_transforms['fashion_mnist'])
        val_dataset = torchvision.datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=data_transforms['fashion_mnist'])     
    elif dataset.name == 'cifar-10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['cifar-10'])
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['cifar-10'])
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

    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    print(images.shape, labels.shape)

    if model.name == 'resnet18':
        ml_model = torchvision.models.resnet18(weights='DEFAULT')
        num_ftrs = ml_model.fc.in_features
        ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'densenet121':
        ml_model = torchvision.models.densenet121(weights='DEFAULT')
        num_ftrs = ml_model.classifier.in_features
        ml_model.classifier = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'alexnet':
        ml_model = torchvision.models.alexnet(weights='DEFAULT')
        num_ftrs = ml_model.classifier[6].in_features
        ml_model.classifier[6] = torch.nn.Linear(num_ftrs, dataset.num_classes)
    elif model.name == 'convnext_base':
        ml_model = torchvision.models.convnext_base(weights='DEFAULT')
        num_ftrs = ml_model.classifier[2].in_features
        ml_model.classifier[2] = torch.nn.Linear(num_ftrs, dataset.num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ml_model, model_history = train_model(ml_model, model, project, train_dataloader, val_dataloader, criterion, optimizer, exp_lr_scheduler, device, num_epochs=NUM_EPOCHS)

    accuracy, precision, recall, f1 = compute_metrics(ml_model, val_dataloader, device)

    history = History(accuracy=accuracy, precision=precision, recall=recall, f1=f1, model_id=model.id)

    db.session.add(history)
    db.session.commit()

    for i in range(len(model_history)):
        epoch = Epoch(epoch=i, train_acc=model_history[i][0], val_acc=model_history[i][1], train_loss=model_history[i][2], val_loss=model_history[i][3], model_id=model.id, history_id=history.id)
        db.session.add(epoch)
    db.session.commit()

    return history.to_dict()


def run_labelling_using_model(app_context, project, dataset, model):
    
    with app_context:
        # initialise model
        if model.name == 'resnet18':
            ml_model = models.resnet18(weights='DEFAULT')
            num_ftrs = ml_model.fc.in_features
            ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)

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

        return instances_updated