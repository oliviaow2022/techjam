import torchvision
import numpy as np
import torch
import os, time
from models import db, History, Epoch, DataInstance
from tempfile import TemporaryDirectory
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from services.S3ImageDataset import s3, S3ImageDataset, download_weights_from_s3

def get_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    return model, in_features


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))

def get_data_loader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def split_train_data(data_loader, batch_size, train_test_split_ratio):
    num_samples = len(data_loader.dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(train_test_split_ratio * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_set = Subset(data_loader.dataset, train_indices)
    val_set = Subset(data_loader.dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=data_loader.collate_fn)
    return train_loader, val_loader

# def train_model(app_context, ml_model, model, project, train_loader, val_loader, optimizer, device, num_epochs):
#     with app_context:
#         since = time.time()
#         model.to(device)

#         # Create a temporary directory to save training checkpoints
#         with TemporaryDirectory() as tempdir:
#             best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
#             torch.save(model.state_dict(), best_model_params_path)
#             best_acc = 0.0
#             history = []
#             for epoch in range(num_epochs):
#                 model.train()
#                 i = 0
#                 for imgs, annotations in train_loader:
#                     i += 1
#                     imgs = list(img.to(device) for img in imgs)
#                     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#                     loss_dict = model(imgs, annotations)
#                     losses = sum(loss for loss in loss_dict.values())

#                     optimizer.zero_grad()
#                     losses.backward()
#                     optimizer.step()

#                     print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len_dataloader}], Loss: {losses.item()}")
#                     if i ==10 :
#                         break
#                 if epoch == 0:
#                     break
#             return model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for imgs, annotations in progress_bar:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        optimizer.zero_grad()
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item() * len(imgs)
        
        progress_bar.set_postfix(loss=losses.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, annotations in dataloader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item() * len(imgs)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_model(app_context, model, model_db, project, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    with app_context:
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            torch.save(model.state_dict(), best_model_params_path)
            best_loss = float('inf')
            history = []

            for epoch in range(num_epochs):
                # Each epoch has a training and validation phase
                train_loss = train_epoch(model, train_loader, optimizer, device)
                scheduler.step()
                val_loss = validate_epoch(model, val_loader, device)

                # Cache metrics for later comparison
                history.append([train_loss, val_loss])
                print(f'Epoch {epoch}/{num_epochs - 1}: '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')

                # Deep copy the model
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), best_model_params_path)

            time_elapsed = time.time() - since
            print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best Val Loss: {best_loss:.4f}')

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
    
def run_training(train_loader, val_loader, app_context, project, dataset, model, num_epochs, train_test_split, batch_size):
    with app_context:
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        print(images.shape, labels.shape)

        if model.name == 'Faster RCNN':
            ml_model, num_ftrs = get_model_instance(num_classes=dataset.num_classes)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # params = [p for p in model.parameters() if p.requires_grad]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.001, momentum=0.9)
    
    ml_model, model_history = train_model(app_context, ml_model, model, project, train_loader, val_loader, optimizer, device, num_epochs)
