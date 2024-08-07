from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import boto3
import torch
import os
from models import Annotation
import torchvision.transforms as transforms

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY')
)


def download_weights_from_s3(bucket_name, key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return obj['Body'].read()

class S3ImageDataset(Dataset):
    def __init__(self, dataframe, bucket, prefix, has_labels=True, transform=None):
        self.dataframe = dataframe
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.dataframe)
    
    def load_image_from_s3(self, key):
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        img = Image.open(BytesIO(obj['Body'].read())).convert('RGB')
        return img

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'data']
        label = self.dataframe.loc[idx, 'labels']
        img_key = f'{self.prefix}/{img_name}'
        
        image = self.load_image_from_s3(img_key)
        
        if self.transform:
            image = self.transform(image)

        if not self.has_labels:
            data_instance_id = self.dataframe.loc[idx, 'id']
            return image, data_instance_id

        # image = torch.tensor(image)
        label = torch.tensor(label)
        
        return image, label
    

class ObjDetDataset(Dataset):
    def __init__(self, annotation_ids, bucket, prefix, label_to_class_mapping):
        self.annotation_ids = annotation_ids # list of annotation_ids
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transforms.ToTensor()
        self.label_to_class_mapping = label_to_class_mapping

    def __len__(self):
        return len(self.annotation_ids)

    def load_image_from_s3(self, key):
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        img = Image.open(BytesIO(obj['Body'].read())).convert('RGB')
        return img 
    
    def __getitem__(self, idx):
        annotation_id = self.annotation_ids[idx]
        annotation = Annotation.query.get_or_404(annotation_id) # get annotation info from db
        annotation.labels = [self.label_to_class_mapping[label] if isinstance(label, str) else label for label in annotation.labels]
        
        img_key = f'{self.prefix}/{annotation.filename}'
        image = self.load_image_from_s3(img_key)
        image = self.transform(image) # Convert image to tensor

        # Convert all values in the annotation dictionary to tensors
        annotation_dict = {}
        annotation_dict["id"] = torch.as_tensor(annotation.id)
        annotation_dict["boxes"] = torch.as_tensor(annotation.boxes)
        annotation_dict["labels"] = torch.as_tensor(annotation.labels)
        annotation_dict["area"] = torch.as_tensor(annotation.area)
        annotation_dict["iscrowd"] = torch.as_tensor(annotation.iscrowd)

        return image, annotation_dict