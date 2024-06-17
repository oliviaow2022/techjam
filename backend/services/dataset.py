from models import Dataset, DataInstance
import pandas as pd

def get_dataframe(dataset_id, return_labelled=True):
    dataset = Dataset.query.get_or_404(dataset_id, description="Dataset ID not found")

    if return_labelled:
        data_instances = DataInstance.query.filter(DataInstance.dataset_id == dataset.id,
                                               DataInstance.labels.isnot(None)).all()
    else:
         data_instances = DataInstance.query.filter(DataInstance.dataset_id == dataset.id,
                                               DataInstance.labels.is_(None)).all()

    data_list = [instance.to_dict() for instance in data_instances]
    df = pd.DataFrame(data_list)
    return df