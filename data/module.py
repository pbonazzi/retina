import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.datasets.ini_30.helper import get_ini_30_dataset
from data.datasets.synthetic_3et.helper import get_3et_dataset

def select_dataset(dataset_name):
    if dataset_name == "ini-30":
        return get_ini_30_dataset
    elif dataset_name == "3et-data":
        return get_3et_dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")

class EyeTrackingDataModule(pl.LightningDataModule):
    def __init__(self, dataset_params, training_params, num_workers=4):
        super().__init__()
        self.dataset_name = dataset_params["dataset_name"]
        self.dataset_params = dataset_params
        self.training_params = training_params
        self.batch_size = training_params["batch_size"]
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Setup for each stage (train, val, test) based on the dataset
        get_dataset_fn = select_dataset(self.dataset_name) 

        self.train_dataset = get_dataset_fn(
            name="train", 
            dataset_params=self.dataset_params, 
            training_params=self.training_params, 
        )
        self.val_dataset = get_dataset_fn(
            name="val", 
            dataset_params=self.dataset_params,
            training_params=self.training_params, 
        )

    def train_dataloader(self, sampler=None):
        return DataLoader(self.train_dataset, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True if sampler is None else False, pin_memory=True, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True, drop_last=True)

    def test_dataloader(self): 
        return None  # Placeholder for test dataloader

