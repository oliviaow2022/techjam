from models import Dataset, DataInstance
import pandas as pd

def get_dataframe(dataset_id, return_all=True):
    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")

    if return_all:
        data_instances = DataInstance.query.filter_by(dataset_id=dataset.id).all()
    else:
        data_instances = DataInstance.query.filter(DataInstance.dataset_id == dataset.id,
                                               DataInstance.labels.isnot(None)).all()

    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df