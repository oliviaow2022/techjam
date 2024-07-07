NUM_CLASSES = 2
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8
NUM_EPOCHS = 3
S3_BUCKET = ''
S3_PREFIX = ''
METADATA_FILE = 'metadata.csv' 
UNLABELED_FILE = 'unlabeled.csv'

from torchvision import transforms

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

from io import BytesIO
from PIL import Image
import boto3
import pandas as pd

s3 = boto3.client('s3')
obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/{METADATA_FILE}')
metadata = pd.read_csv(BytesIO(obj['Body'].read()))

obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{S3_PREFIX}/{UNLABELED_FILE}')
unlabeled = pd.read_csv(BytesIO(obj['Body'].read()))

# Define a function to load images from S3
def load_image_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    img = Image.open(BytesIO(obj['Body'].read())).convert('RGB')
    return img

from torch.utils.data import Dataset

class S3ImageDataset(Dataset):
    def __init__(self, metadata, bucket, prefix, has_labels=True, transform=None):
        self.metadata = metadata
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 0]
        if self.has_labels:
            label = self.metadata.iloc[idx, 1]
        img_key = f'{self.prefix}/{img_name}'
        
        image = load_image_from_s3(self.bucket, img_key)
        
        if self.transform:
            image = self.transform(image)
        
        if self.has_labels: 
            return image, label
        
        return image, img_name
    
from torch.utils.data import random_split

dataset = S3ImageDataset(metadata, S3_BUCKET, S3_PREFIX)

train_size = int(TRAIN_TEST_SPLIT * len(dataset)) 
val_size = len(dataset) - train_size 

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

import torch
import torchvision.models as models

model_conv = models.ConvNext Base(weights='DEFAULT')
for param in model_conv.parameters(): param.requires_grad = False
model_conv.classifier[2] = torch.nn.Linear(
    in_features=model_conv.classifier[2].in_features,
    out_features=NUM_CLASSES
)

from tempfile import TemporaryDirectory
import os, time

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    
    for inputs, labels in dataloader:
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        history = []

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            train_loss, train_corrects = train_epoch(model, train_dataloader, criterion, optimizer, device)
            scheduler.step()
            val_loss, val_corrects = validate_epoch(model, val_dataloader, criterion, device)

            train_loss /= train_size
            train_acc = train_corrects / train_size
            val_loss /= val_size
            val_acc = val_corrects / val_size

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
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, history

from torch.optim import lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()
optimizer_conv = torch.optim.Adam(model_conv.classifier[2].parameters())
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv, model_conv_history = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, NUM_EPOCHS)

def get_predictions(model, dataloader):
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for images, filenames in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            confidence_levels = torch.nn.functional.softmax(outputs, dim=1)
            entropy = -torch.sum(confidence_levels * torch.log2(confidence_levels))
            print(outputs)
            print(confidence_levels)
            print(entropy)
            break
            for filename, confidence in zip(filenames, confidence_levels):
                all_predictions.append((filename, predicted.item(), confidence))

    return all_predictions

unlabeled_ds = S3ImageDataset(unlabeled, S3_BUCKET, S3_PREFIX, has_labels=None)
unlabeled_ds.transform = data_transforms['val']
unlabeled_dataloader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True)

predictions = get_predictions(model_conv, unlabeled_dataloader)
""" for filename, predicted_class, confidence in predictions:
    print(f'Image: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence[predicted_class]:.4f}')
 """