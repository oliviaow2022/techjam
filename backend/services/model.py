import torch
import os, time
from tempfile import TemporaryDirectory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from S3ImageDataset import s3, S3ImageDataset, download_weights_from_s3
from models import db, History, Epoch, DataInstance
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torchvision
from torch.optim import lr_scheduler
from services.dataset import get_dataframe
from io import BytesIO

import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, TensorDataset
from torch import nn, optim
import torchvision.models as models

from PIL import Image
import boto3
from io import BytesIO

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


def train_model(app_context, model, model_db, project, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, NUM_EPOCHS):

    with app_context:
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0
            history = []

            for epoch in range(NUM_EPOCHS):
                # Each epoch has a training and validation phase
                train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
                scheduler.step()
                val_loss, val_acc = validate_epoch(model, val_dataloader, criterion, device)

                # Cache metrics for later comparison
                history.append([train_acc, val_acc, train_loss, val_loss])
                print(f'Epoch {epoch}/{NUM_EPOCHS - 1}: '
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
            
            print(model_db)
            # save model to db
            model_db.saved = model_path
            db.session.add(model_db)
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


def run_training(app_context, project, dataset, model, NUM_EPOCHS, TRAIN_TEST_SPLIT, BATCH_SIZE):

    with app_context:
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

        if model.name == 'ResNet-18':
            ml_model = torchvision.models.resnet18(weights='DEFAULT')
            num_ftrs = ml_model.fc.in_features
            ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'DenseNet-121':
            ml_model = torchvision.models.densenet121(weights='DEFAULT')
            num_ftrs = ml_model.classifier.in_features
            ml_model.classifier = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'AlexNet':
            ml_model = torchvision.models.alexnet(weights='DEFAULT')
            num_ftrs = ml_model.classifier[6].in_features
            ml_model.classifier[6] = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'ConvNext Base':
            ml_model = torchvision.models.convnext_base(weights='DEFAULT')
            num_ftrs = ml_model.classifier[2].in_features
            ml_model.classifier[2] = torch.nn.Linear(num_ftrs, dataset.num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        ml_model, model_history = train_model(app_context, ml_model, model, project, train_dataloader, val_dataloader, criterion, optimizer, exp_lr_scheduler, device, NUM_EPOCHS)

        accuracy, precision, recall, f1 = compute_metrics(ml_model, val_dataloader, device)

        history = History(accuracy=accuracy, precision=precision, recall=recall, f1=f1, model_id=model.id)

        db.session.add(history)
        db.session.commit()

        for i in range(len(model_history)):
            epoch = Epoch(epoch=i, train_acc=model_history[i][0], val_acc=model_history[i][1], train_loss=model_history[i][2], val_loss=model_history[i][3], model_id=model.id, history_id=history.id)
            db.session.add(epoch)
        db.session.commit()

        print(history.to_dict())

        run_labelling_using_model(app_context, project, dataset, model)


def run_labelling_using_model(app_context, project, dataset, model):
    
    with app_context:
        # initialise model
        if model.name == 'ResNet-18':
            ml_model = torchvision.models.resnet18(weights='DEFAULT')
            num_ftrs = ml_model.fc.in_features
            ml_model.fc = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'DenseNet-121':
            ml_model = torchvision.models.densenet121(weights='DEFAULT')
            num_ftrs = ml_model.classifier.in_features
            ml_model.classifier = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'AlexNet':
            ml_model = torchvision.models.alexnet(weights='DEFAULT')
            num_ftrs = ml_model.classifier[6].in_features
            ml_model.classifier[6] = torch.nn.Linear(num_ftrs, dataset.num_classes)
        elif model.name == 'ConvNext Base':
            ml_model = torchvision.models.convnext_base(weights='DEFAULT')
            num_ftrs = ml_model.classifier[2].in_features
            ml_model.classifier[2] = torch.nn.Linear(num_ftrs, dataset.num_classes)

        # load in weights
        model_weights = download_weights_from_s3(project.bucket, model.saved)
        ml_model.load_state_dict(torch.load(BytesIO(model_weights)))
        ml_model.eval()

        # only get datainstances with no labels
        df = get_dataframe(dataset.id, return_labelled=False)

        s3_dataset = S3ImageDataset(df, project.bucket, project.prefix, has_labels=False)
        s3_dataset.transform = data_transforms['image_val']

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
                    # data_instance.labels = predicted[index].item()
                    db.session.commit()
                    instances_updated += 1

        return instances_updated

class UnlabeledDataset(Dataset):
    def __init__(self, bucket_name, prefix, transform=None):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.s3 = boto3.client('s3', aws_access_key_id="AKIA6GBMD5NHN6NOW77I", aws_secret_access_key="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")
        self.images = self._get_image_keys()

    def _get_image_keys(self):
        """Retrieve the list of image keys (paths) from the S3 bucket."""
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        keys = [content['Key'] for content in response.get('Contents', [])]
        return keys

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_key = self.images[idx]
        response = self.s3.get_object(Bucket=self.bucket_name, Key=img_key)
        image = Image.open(BytesIO(response['Body'].read())).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class TorchModelWrapper:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, y, num_epochs=1):
        self.model.train()
        X = torch.stack([x for x in X])  # Convert to tensor
        y = torch.tensor(y)  # Convert to tensor

        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
            
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loader)}")

    def predict_proba(self, X):
        self.model.eval()
        # preds = []
        # proba = []
        tensors = []
        with torch.no_grad():
            if isinstance(X, list):
                for x in X:
                    if len(x.shape) == 4:
                        x=x[0]
                    # print("Edited x", x.shape)
                    # # x=torch.unbind(x, dim=0)
                    # # print("X edit", x)
                    if len(x.shape)==3 and x.shape==torch.Size([3, 224, 224]):
                        tensors.append(x)
                # X = [x[0] for x in X if len(x.shape)==3 and x.shape==torch.Size([3, 224, 224])]
                if tensors:
                    X=torch.stack(tensors)
                else:
                    raise ValueError("No valid tensors found in the input list")
            if len(X.shape) == 3:
                X = X.unsqueeze(0)
            outputs = self.model(X)

            return outputs.cpu().numpy()


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define transform to resize and normalize images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fixed size (e.g., 224x224)
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def dataloader_to_tensor_list(dataloader):
    x = []
    ids = []
    for images, data_instance_ids in dataloader:
        # for img in images:
        x.extend(images)
        ids.extend(data_instance_ids)
    return x, ids

def run_labelling_with_model(app_context, dataset_details, dataset, project):
    with app_context:
        resnet50 = models.resnet50(pretrained=True)
        print("nihao2")

        # Modify the final fully connected layer for multi-label classification
        num_features = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, dataset_details.num_classes),  # Adjust output size based on number of classes
            nn.Sigmoid()  # Sigmoid activation for multi-label classification
        )

        # Freeze early layers if necessary (optional)
        # for param in resnet50.parameters():
        #     param.requires_grad = False

        # Initialize the model, loss function, and optimizer
        model = resnet50
        criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-label classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        df = get_dataframe(dataset.id, return_labelled=False)

        s3_dataset = S3ImageDataset(df, project.bucket, project.prefix, has_labels=False)
        # print(s3_dataset)
        # unlabelled_dataset = UnlabeledDataset(project.bucket, project.prefix, transform=transform)
        s3_dataset.transform = transform

        unlabelled_loader = DataLoader(s3_dataset, batch_size=32, shuffle=True)
        # unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=32, shuffle=True)

        learner = ActiveLearner(
            estimator=TorchModelWrapper(model, optimizer, criterion),
            query_strategy=uncertainty_sampling
        )

        data_instances = []
        for i in range(3):
            print(i)
            X_pool, ids = dataloader_to_tensor_list(unlabelled_loader)
            query_idx, _ = learner.query(X_pool)
            print(query_idx)
        
            # selected_instances = [ids[i] for i in query_idx]
            instances_updated = 0

            # Calculate confidence and entropy for selected instances
            model.eval()
            with torch.no_grad():
                for idx in query_idx:
                    image = X_pool[idx].unsqueeze(0)  # Add batch dimension
                    output = model(image)
                    confidence_levels = torch.nn.functional.softmax(output, dim=1)
                    entropy = -torch.sum(confidence_levels * torch.log2(confidence_levels + 1e-10), dim=1)
                    
                for index, instance_id in enumerate(ids):
                    data_instance = DataInstance.query.get_or_404(instance_id.item(), description='DataInstance ID not found')
                    print(data_instance)
                    data_instance.entropy = entropy[index].item()
                    # data_instance.labels = predicted[index].item()
                    db.session.commit()
                    instances_updated += 1

        return instances_updated

