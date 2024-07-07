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
import os
# import boto3
# import pandas as pd
# from io import BytesIO

class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, img) for img in os.listdir(root) if os.path.isfile(os.path.join(root, img))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
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
        with torch.no_grad():
            if isinstance(X, list):
                X=torch.stack(X)
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

print("hello")
# Load dataset
labelled_dataset = ImageFolder(root='C:/Users/natha/Documents/TikTokTechJam/multilabeldata/car/labelled', transform=transform)
unlabelled_dataset = UnlabeledDataset(root='C:/Users/natha/Documents/TikTokTechJam/multilabeldata/car/unlabelled', transform=transform)

# Split data into initial labeled set and unlabeled pool
train_size = 100
indices = list(range(len(labelled_dataset)))
np.random.shuffle(indices)
train_idx = indices[:train_size]

train_sampler = SubsetRandomSampler(train_idx)

train_loader = DataLoader(labelled_dataset, sampler=train_sampler, batch_size=32)
unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=32, shuffle=True)

resnet50 = models.resnet50(pretrained=True)
print("nihao")

# Modify the final fully connected layer for multi-label classification
num_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(labelled_dataset.classes)),  # Adjust output size based on number of classes
    nn.Softmax(dim=1)  # Sigmoid activation for multi-label classification
)

# Freeze early layers if necessary (optional)
# for param in resnet50.parameters():
#     param.requires_grad = False

# Initialize the model, loss function, and optimizer
model = resnet50
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def dataloader_to_tensor_list(dataloader):
    x = []
    for images in dataloader:
        x.extend(images)
    return x

# Active learning setup with modAL
learner = ActiveLearner(
    estimator=TorchModelWrapper(model, optimizer, criterion),
    query_strategy=uncertainty_sampling
)

# Number of queries
n_queries = 10
print(n_queries)
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    model.train()
    total = 0
    correct= 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get the predicted classes
            total += labels.size(0)  # Increase total by batch size
            correct += (predicted == labels).sum().item()  # Count correct predictions            
            # predicted = (outputs > 0.5).float()  # Use 0.5 threshold for binary classification
            accuracy = correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        print(f"Accuracy: {accuracy}")

    return model

# trained_model = train_model(model, criterion, optimizer, train_loader)
# torch.save(trained_model.state_dict(), 'trained_model.pth')

# Perform active learning loop
for i in range(n_queries):
    print("Query", i)

    # Query for the most uncertain instances
    X_pool = dataloader_to_tensor_list(unlabelled_loader)
    query_idx, _ = learner.query(X_pool)
    print(query_idx)

    # Simulate labeling for demo purposes (in real case, you'd label these manually)
    X_new = [unlabelled_dataset[i][0] for i in query_idx]
    y_new = [labelled_dataset[i][1] for i in query_idx]
    
    #Teach learner
    # learner.teach(X_new, y_new)

    #update train daaset and data loader
    train_idx = np.append(train_idx, query_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(labelled_dataset, sampler=train_sampler, batch_size=32)

    # Remove queried instances from pool
    remaining_unlabeled_idx = np.setdiff1d(np.arange(len(unlabelled_dataset)), query_idx)
    unlabeled_loader = DataLoader(unlabelled_dataset, sampler=SubsetRandomSampler(remaining_unlabeled_idx), batch_size=16)

    # Retrain the model with the expanded training set
    trained_model = train_model(trained_model, criterion, optimizer, train_loader)

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted classes
            total += labels.size(0)  # Increase total by batch size
            correct += (predicted == labels).sum().item()  # Count correct predictions            
            # predicted = (outputs > 0.5).float()  # Use 0.5 threshold for binary classification
            # total += labels.size(0)
            # total += labels.numel()
            # correct += ((predicted == labels).sum(dim=1) == labels.size(1)).sum().item()
            confidence_levels = torch.nn.functional.softmax(outputs, dim=1)
            entropy = -torch.sum(confidence_levels * torch.log2(confidence_levels))

            accuracy = correct / total
    return entropy, accuracy

# entropy, model_accuracy = evaluate_model(trained_model, DataLoader(labelled_dataset, batch_size=32))

# print(f"Accuracy after active learning: {model_accuracy}")
# print(f"Entropy after active learning: {entropy}")