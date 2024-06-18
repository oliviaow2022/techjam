from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import boto3
import torch

s3 = boto3.client('s3')

class S3ImageDataset(Dataset):
    def __init__(self, dataframe, bucket, prefix, transform=None):
        self.dataframe = dataframe
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform

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

        # image = torch.tensor(image)
        label = torch.tensor(label)
        
        return image, label