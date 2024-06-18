import torch
import os, time
from tempfile import TemporaryDirectory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from S3ImageDataset import s3
from models import db

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