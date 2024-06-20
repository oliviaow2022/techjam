from models import Dataset, DataInstance
import pandas as pd

def get_dataframe(dataset_id, return_labelled=True):
    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")

    if return_labelled:
        # return data with labels for training
        data_instances = DataInstance.query.filter(DataInstance.dataset_id == dataset.id,
                                               DataInstance.labels.isnot(None)).all()
    else:
         # return data without manually labeled labels for training
         data_instances = DataInstance.query.filter_by(dataset_id=dataset.id, manually_processed=False).all()

    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df


def get_dataset_config(dataset_name, num_classes, class_to_label_mapping):
    if dataset_name == 'fashion-mnist':
        num_classes = 10
        class_to_label_mapping = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle Boot'
        }
    elif dataset_name == 'cifar-10':
        num_classes = 10
        class_to_label_mapping = {
            0: 'plane',
            1: 'car',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
        
    return num_classes, class_to_label_mapping