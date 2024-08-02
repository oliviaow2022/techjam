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

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, annotations in tqdm(train_loader, desc="Training"):
        imgs = list(img.to(device) for img in images)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    return running_loss / len(train_loader)

from torchvision.ops import box_iou

def validate_epoch(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_annotations = []
    
    with torch.no_grad():
        for images, annotations in tqdm(val_loader, desc="Validating"):
            images = [image.to(device) for image in images]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            outputs = model(images)
            all_predictions.extend(outputs)
            all_annotations.extend(annotations)

    tp, fp, fn = 0, 0, 0
            
    for pred, annot in zip(all_predictions, all_annotations):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        
        true_boxes = annot['boxes']
        true_labels = annot['labels']

        if len(pred_boxes) == 0:
            fn += len(true_boxes)
            continue
        
        if len(true_boxes) == 0:
            fp += len(pred_boxes)
            continue

        ious = box_iou(pred_boxes, true_boxes)
        
        for j in range(len(pred_boxes)):
            # find the ground truth bbox with the highest IOU with the current predicted bbox
            max_iou, max_idx = torch.max(ious[j], dim=0)
            if max_iou > iou_threshold:
                if pred_labels[j] == true_labels[max_idx]:
                    tp += 1
                else:
                    fp += 1
                # Ensure that this true box is not matched again
                ious[:, max_idx] = 0
            else:
                fp += 1
        
        # Calculate false negatives
        matched_true_boxes = torch.sum(ious > iou_threshold, dim=0)
        fn += len(true_boxes) - torch.sum(matched_true_boxes > 0).item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

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
                precision, recall = validate_epoch(model, val_loader, device)

                # Cache metrics for later comparison
                history.append([train_loss, precision, recall])
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(model.state_dict(), best_model_params_path)

            time_elapsed = time.time() - since
            print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best Train Loss: {best_loss:.4f}')

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
    
def run_training(self, train_loader, val_loader, app_context, project, dataset, model, num_epochs, train_test_split, batch_size):
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

    for i, (train_loss, precision, recall) in enumerate(model_history):
        epoch = Epoch(epoch=i, train_loss=train_loss, precision=precision, recall=recall, model_id=model.id)
        db.session.add(epoch)
    db.session.commit()
    print('Training complete and data saved to the database.')