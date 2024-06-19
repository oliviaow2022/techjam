from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import boto3
import torch

s3 = boto3.client('s3')

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