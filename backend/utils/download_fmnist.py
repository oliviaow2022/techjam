import torchvision
import uuid
from S3ImageDataset import s3
import os
from models import db, DataInstance
from services.model import data_transforms
from tqdm import tqdm

from app import app
with app.app_context():

    S3_BUCKET = os.getenv('S3_BUCKET')
    S3_PREFIX = 'fashion-mnist'

    train_dataset = torchvision.datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=data_transforms['fashion_mnist'])
    # val_dataset = torchvision.datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=data_transforms['fashion_mnist'])     

    data = []

    os.makedirs('tmp', exist_ok=True)
    for image, label in tqdm(train_dataset):
        file_name = f'{uuid.uuid4()}.png'
        local_file_path = os.path.join('/tmp', file_name)

        image_pil = torchvision.transforms.functional.to_pil_image(image)
        image_pil.save(local_file_path)

        s3_file_path = f'{S3_PREFIX}/{file_name}'
        s3.upload_file(local_file_path, S3_BUCKET, s3_file_path)

        data_instance = DataInstance(data=file_name, dataset_id=2, labels=str(label))
        db.session.add(data_instance)
        db.session.commit()
        
