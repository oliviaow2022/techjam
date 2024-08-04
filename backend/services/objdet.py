import torchvision
import numpy as np
import torch
import os, time
import json

from models import db, History, Epoch, Model, Annotation
from tempfile import TemporaryDirectory
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from services.S3ImageDataset import s3, ObjDetDataset
from torchvision.ops import box_iou
from tqdm import tqdm
from celery import shared_task


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

    return model

def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))

def split_train_data(data_loader, train_test_split_ratio):
    num_samples = len(data_loader.dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(train_test_split_ratio * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_set = Subset(data_loader.dataset, train_indices)
    val_set = Subset(data_loader.dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=data_loader.batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(val_set, batch_size=data_loader.batch_size, shuffle=False, collate_fn=data_loader.collate_fn)
    return train_loader, val_loader

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


def train_model(self, model, model_dict, project_dict, train_loader, val_loader, optimizer, device, num_epochs=1):

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_loss = float('inf')
        history = []

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            precision, recall = validate_epoch(model, val_loader, device)

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), best_model_params_path)

            history.append([train_loss, precision, recall])
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            # update task state
            self.update_state(task_id=self.request.id, state="PROGRESS", meta={'epoch': epoch, 'num_epochs': num_epochs})

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Loss: {best_loss:.4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        # upload model to s3
        model_path = f"{project_dict['prefix']}/{model_dict['name']}_{model_dict['id']}.pth"
        s3.upload_file(best_model_params_path, project_dict['bucket'], model_path)
        
        print(model_dict)
        # save model to db
        model_db = Model.query.get_or_404(model_dict['id'])
        model_db.saved = model_path
        db.session.add(model_db)
        db.session.commit()
        print('model saved to', model_path)

    return model, history


def get_predictions(model, data_loader, device):
    model.eval()

    instances_updated = 0

    with torch.no_grad():
        for images, annotations in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                avg_confidence = output['scores'].mean().item()
                if torch.isnan(torch.tensor(avg_confidence)): # if no bounding boxes, scores is empty, mean as nan
                    avg_confidence = 0
                    
                print(output) 
                print('annotations', annotations[i])
                print('avg_confidence', avg_confidence)

                # update confidence in db
                annotation = Annotation.query.get_or_404(annotations[i]['id'].item())
                annotation.confidence = avg_confidence
                db.session.commit()
                instances_updated += 1
    
    return instances_updated


@shared_task(bind=True)
def run_training(self, annotation_ids, project_dict, dataset_dict, model_dict, history_dict, NUM_EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT):
    # reverse class_to_label_mapping
    print(dataset_dict['class_to_label_mapping'])
    label_to_class_mapping = {v: int(k) for k, v in json.loads(dataset_dict['class_to_label_mapping']).items()}
    print(label_to_class_mapping)

    # set up dataset
    labelled_dataset = ObjDetDataset(annotation_ids, project_dict['bucket'], project_dict['prefix'], label_to_class_mapping)
    labelled_data_loader = torch.utils.data.DataLoader(labelled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=collate_fn)
    train_loader, val_loader = split_train_data(labelled_data_loader, TRAIN_TEST_SPLIT)

    # set up model, optimizer and device
    ml_model = get_model_instance(num_classes=dataset_dict['num_classes'])
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=1e-4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ml_model, training_history = train_model(self, ml_model, model_dict, project_dict, train_loader, val_loader, optimizer, device, NUM_EPOCHS)

    precision, recall = validate_epoch(ml_model, val_loader, device)
    history_db = History.query.get_or_404(history_dict['id'])
    history_db.precision = precision
    history_db.recall = recall
    db.session.add(history_db)
    db.session.commit()

    for i in range(len(training_history)):
        epoch = Epoch(epoch=i, train_loss=training_history[i][0], precision=training_history[i][1], recall=training_history[i][2], model_id=model_dict['id'], history_id=history_dict['id'])
        db.session.add(epoch)
        db.session.commit()

    get_predictions(ml_model, val_loader, device)

    