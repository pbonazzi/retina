"""
This is a file script used for loading the dataloader
"""
import pdb
import torch
from PIL import Image
import numpy as np

# custom
from data.utils import load_yaml_config
from data.transforms.helper import get_transforms
from data.datasets.ini_30.ini_30_dataset import Ini30Dataset 

def get_ini_30_dataset(name, training_params, dataset_params):
    """
    Create and return a Dataset from the Ini30Dataset.

    Parameters:
        data_dir (str): The directory path where the dataset is located.
        batch_size (int): The batch size used in the DataLoader.
        num_bins (int): The number of bins used for transformation.
        idxs (List[int]): A list of experiment indices to include in the dataset.

    Returns:
        Dataset: The Dataset object from the Ini30Dataset.
    """

    input_transforms, target_transforms = get_transforms(dataset_params, training_params)

    dataset = Ini30Dataset( 
        dataset_params=dataset_params, 
        transform=input_transforms,
        target_transform=target_transforms, 
        list_experiments=get_indexes(name, dataset_params["ini30_val_idx"])
    )

    return dataset

def get_indexes(name, val_idx): 

    if name=="val":
        return val_idx

    elif name=="train":
        all_idxs = list(range(0, 30)) 
        for idx in val_idx:
            all_idxs.remove(idx) 
        return all_idxs 

if __name__ == "__main__":

    default_params = load_yaml_config("configs/default.yaml")
    training_params = default_params["training_params"]
    dataset_params = default_params["dataset_params"] 

    dataset = get_ini_30_dataset("train", training_params, dataset_params)

    # plots
    events =  dataset[0][0]
    num_bins, channels, height, width = events.shape
    
    if channels == 1: 
        events = events.repeat(1,3,1,1)
    else:
        events = torch.cat([events[:, :1, ...], 
                            torch.zeros((num_bins, 1, height, width)).to(events.device), 
                            events[:, 1:, ...]], dim=1)

    events = events.transpose(1,2).transpose(2,3)
    events = events.cpu().detach().numpy()

    if channels == 2: 
        zero_indices = (events == [0, 0, 0]).all(axis=-1)
        events[zero_indices] = [1, 1, 1] 

    image = Image.fromarray((events[0]* 255).astype(np.uint8))
    image.save("output_image_0.png")

