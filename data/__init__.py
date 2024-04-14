"""
This is a file script used for loading the dataloader
"""
import pdb
import torch
from tonic.transforms import Compose, ToFrame, MergePolarities, EventDrop, RandomFlipPolarity, Decimation, Denoise, CenterCrop, Downsample
from torch.utils.data import DataLoader
from typing import List, Tuple, Union

# custom
from data.inivation_dataset import EyeTrackingInivationDataset
from data.transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP, Downscale

def get_dataloader(name, training_params, dataset_params, device, shuffle) -> DataLoader:
    """
    Create and return a DataLoader for the EyeTrackingInivationDataset.

    Parameters:
        data_dir (str): The directory path where the dataset is located.
        batch_size (int): The batch size used in the DataLoader.
        num_bins (int): The number of bins used for transformation.
        idxs (List[int]): A list of experiment indices to include in the dataset.

    Returns:
        DataLoader: The DataLoader object for the EyeTrackingInivationDataset.
    """

    input_transforms, target_transforms = get_transforms(dataset_params, training_params)

    dataset = EyeTrackingInivationDataset(
        training_params=training_params,
        dataset_params=dataset_params,
        shuffle=shuffle,
        transform=input_transforms,
        target_transform=target_transforms, 
        list_experiments=training_params[name+"_idxs"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_params["batch_size"],
        drop_last=True,
        shuffle=shuffle,
        generator=torch.Generator(device=device),
        num_workers=0, 
    )

    return dataloader


def get_transforms(dataset_params, training_params, augmentations: bool = False) -> Tuple[Compose, FromPupilCenterToBoundingBox]:
    """
    Get input and target transforms for the EyeTrackingInivationDataset.

    Parameters:
        num_bins (int): The number of bins used for transformation.
        augmentations (bool, optional): If True, apply data augmentations. Defaults to False.

    Returns:
        Tuple[Compose, FromPupilCenterToBoundingBox]: A tuple containing the input and target transforms.
    """
    sensor_size = (dataset_params["img_height"], dataset_params["img_width"], dataset_params["input_channel"])

    target_transforms = FromPupilCenterToBoundingBox(   yolo_loss=training_params["yolo_loss"], 
                                                        focal_loss=training_params["focal_loss"],
                                                        bbox_w=training_params["bbox_w"],
                                                        SxS_Grid=training_params["SxS_Grid"],
                                                        num_classes=training_params["num_classes"],
                                                        num_boxes=training_params["num_boxes"],
                                                        synthetic_dataset=dataset_params["synthetic_dataset"],
                                                        image_size=(dataset_params["img_width"], dataset_params["img_height"]),
                                                        num_bins=dataset_params["num_bins"]) 

    input_transforms = [AedatEventsToXYTP()]
    if not dataset_params["synthetic_dataset"] and (dataset_params["img_width"] != 640 or dataset_params["img_height"] != 480):
        input_transforms.append(CenterCrop(sensor_size=(640, 480), size=(512, 512))) 
        input_transforms.append(Downscale())

    if dataset_params["pre_decimate"]:
        input_transforms.append(Decimation(dataset_params["pre_decimate_factor"]))
        
    if dataset_params["denoise_evs"]:
        input_transforms.append(Denoise(filter_time=dataset_params["filter_time"]))
    
    # if dataset_params["random_flip"] == 1:
    #     input_transforms.append(RandomFlipPolarity())

    # if dataset_params["event_drop"]:
    #     input_transforms.append(EventDrop(sensor_size=sensor_size))
        
    if dataset_params["input_channel"] == 1:
        input_transforms.append(MergePolarities())
    
    input_transforms.append(ToFrame(sensor_size=sensor_size,   
                                    n_event_bins=dataset_params["num_bins"]
                                    ))
    input_transforms = Compose(input_transforms)
    return input_transforms, target_transforms


def get_indexes(val_idx=1, remove_experiments=True, overfit=False):
    num_exp = 24 if remove_experiments else 30
    train_val_idxs = list(range(0, num_exp))
    #random.shuffle(train_val_idxs)
    train_val_idxs.remove(val_idx)
    train_idxs = train_val_idxs # 0, 
    val_idxs = [val_idx]  
    if overfit:  train_idxs, val_idxs = train_idxs[:1],  val_idxs[-1:]
    return train_idxs, val_idxs
