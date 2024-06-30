import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
from pycocotools.coco import COCO
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
# from modAL import Dataset
from typing import Tuple
import numpy as np

# # AWS S3 Configuration
# import boto3
# s3_client = boto3.client('s3')
# BUCKET_NAME = 'your-s3-bucket-name'

# Unlabeled dataset class
class UnlabelledDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms or T.ToTensor()
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        image = self.transforms(image)
        return image


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


def train(model, train_loader, device, num_epochs=5):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    len_dataloader = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations in train_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len_dataloader}], Loss: {losses.item()}")
            if i ==10 :
                break
        if epoch == 0:
            break
    return model

class Wrapper:
    def __init__(self, model):
        self.model = model
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([x.to(device) for x in X])  # Ensure each image in X is converted and passed as a list
            probs = [output['scores'].cpu().numpy() for output in outputs]
        return probs
# # Function to extract data from DataLoader
# def extract_data(dataloader):
#     X = []
#     for images in dataloader:
#         X.extend(images)
#     return X

# Simulate data labeling
def label_data(indices):
    labeled_data = []
    for idx in indices:
        # In a real application, this is where you'd present the data to the annotator
        image = unlabeled_dataset[idx]
        target = {"boxes": [], "labels": []}  # Placeholder for the annotation process
        # Simulate the annotation process (in practice, this would be user-provided)
        labeled_data.append((image, target))
    return labeled_data

# Function to retrain the model
def retrain_model(model, train_loader, new_data, num_epochs=1):
    # Add the newly labeled data to the training set
    new_images, new_targets = zip(*new_data)
    train_loader.dataset += list(zip(new_images, new_targets))

    # Retrain the model
    model = train(model, train_loader, num_epochs)
    return model

# Function to iterate the active learning process
def active_learning_iteration(learner, X_unlabeled, n_queries=10):
    query_idx, _ = learner.query(X_unlabeled, n_instances=n_queries)
    print("ABCDJDKEWNFJKNFKJWNEKFJN")
    new_data = label_data(query_idx)

    # Add the new data to the learner
    X_new, y_new = zip(*new_data)
    learner.teach(X_new, y_new)

    return new_data


labelled_data_loader = train_data_loader()

# Split labeled dataset into train and validation
train_loader, val_loader = split_train_data(labelled_data_loader)

# train the initial model based on labelled data
num_classes = 5  # Example: 5 classes
model = get_model_instance_segmentation(num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_model = train(model, val_loader, device=device)

# # Define your active learner with uncertainty sampling
learner = ActiveLearner(
    estimator=train_model,  # trained R-CNN model
    X_training=None,  # Initial labeled dataset (if available)
    y_training=None,  # Initial labels (if available)
    query_strategy=uncertainty_sampling,  # Active learning strategy
)

# Create a dataloader for the unlabeled dataset
unlabeled_dataset = UnlabelledDataset(root_dir="data/unlabelled_data")
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=2, shuffle=False)

model = train_model
# Run active learning iterations
n_iterations = 5
for iteration in range(n_iterations):
    new_data = active_learning_iteration(learner, unlabeled_loader)
    # model = retrain_model(model, train_loader, new_data)

# while learner.query_strategy(train_loader.dataset):
#     query_idx, query_inst = learner.query(train_loader.dataset)
#     # Loop through each instance to manually annotate
#     for inst in query_inst:
#         img, annotations = inst

#         # Display the image for annotation (you can replace this with your preferred annotation tool)
#         img_pil = torchvision.transforms.ToPILImage()(img.squeeze(0))
#         img_pil.show()

#         # Manually annotate and update dataset
#         print(f"Annotate image: {annotations['image_id']}")
#         print("Enter annotation details (xmin, ymin, xmax, ymax, label):")
#         xmin, ymin, xmax, ymax, label = map(int, input().split())

#         annotated_annotations = {
#             "image_id": annotations["image_id"],
#             "file_name": annotations["file_name"],
#             "height": annotations["height"],
#             "width": annotations["width"],
#             "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO-style bounding box format
#             "area": (xmax - xmin) * (ymax - ymin),  # Area calculation
#             "category_id": label,  # Assuming a category ID or label
#             "iscrowd": 0,  # Assuming not crowd
#         }




    

# After annotation loop, retrain the model with the updated dataset
# Example:
# learner.fit(dataset.X_training, dataset.y_training)
