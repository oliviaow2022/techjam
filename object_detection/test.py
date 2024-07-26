# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install pycocotools 

import torch
import os
from pycocotools.coco import COCO
from PIL import Image
from typing import Tuple

class Cppe5(torch.utils.data.Dataset):
    def __init__(self, root: str, annotation: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")  # Convert to RGB

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        annotations = {}
        annotations["boxes"] = boxes
        annotations["labels"] = labels
        annotations["image_id"] = img_id
        annotations["area"] = areas
        annotations["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, annotations

    def __len__(self) -> int:
        return len(self.ids)


import torchvision

def tensor_transform() -> torchvision.transforms.Compose:
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))
    
def train_data_loader(
    train_batch_size: int = 1,
    train_data_dir: str = "data/images",
    train_annotation_file: str = "data/annotations/train.json",
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:

    cppe5 = Cppe5(
        root=train_data_dir,
        annotation=train_annotation_file,
        transforms=tensor_transform(),
    )

    return torch.utils.data.DataLoader(
        cppe5,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def test_data_loader(
    train_batch_size: int = 1,
    train_data_dir: str = "data/images",
    train_annotation_file: str = "data/annotations/test.json",
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:

    cppe5 = Cppe5(
        root=train_data_dir,
        annotation=train_annotation_file,
        transforms=tensor_transform(),
    )

    return torch.utils.data.DataLoader(
        cppe5,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def get_model_instance_segmentation(num_classes):
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

from torch.utils.data import DataLoader, Subset
import numpy as np

def split_train_data(data_loader):
    num_samples = len(data_loader.dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_set = Subset(data_loader.dataset, train_indices)
    val_set = Subset(data_loader.dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=data_loader.batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(val_set, batch_size=data_loader.batch_size, shuffle=False, collate_fn=data_loader.collate_fn)
    return train_loader, val_loader

def truncate_data(data_loader):
    train_set = Subset(data_loader.dataset, list(range(10)))
    val_set = Subset(data_loader.dataset, list(range(10, 15)))
    train_loader = DataLoader(train_set, batch_size=data_loader.batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
    val_loader = DataLoader(val_set, batch_size=data_loader.batch_size, shuffle=False, collate_fn=data_loader.collate_fn)
    return train_loader, val_loader

from tqdm import tqdm

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

def validate_epoch(model, val_loader, iou_threshold=0.5):
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


def train_model(model, train_loader, val_loader, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        precision, recall = validate_epoch(model, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


# set up dataset
labelled_data_loader = train_data_loader()
train_loader, val_loader = split_train_data(labelled_data_loader)

# set up model, optimizer and device
num_classes = 5
model = get_model_instance_segmentation(num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# truncate data for easier testing
train_loader_truncated, val_loader_truncated = truncate_data(labelled_data_loader)

# view contents of dataloader 
for imgs, annotations in train_loader_truncated:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)
    break

train_model(model, train_loader_truncated, val_loader_truncated, optimizer, num_epochs=1)

def get_predictions(model, data_loader):
    model.eval()

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                avg_confidence = output['scores'].mean().item()
                print(avg_confidence)
            break

get_predictions(model, val_loader_truncated)
