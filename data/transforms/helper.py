from typing import List, Tuple, Union
from tonic.transforms import Compose, ToFrame, MergePolarities, EventDrop, RandomFlipPolarity, Decimation, Denoise, CenterCrop, Downsample
from .transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP, Downscale


def get_transforms(dataset_params, training_params, augmentations: bool = False) -> Tuple[Compose, FromPupilCenterToBoundingBox]:
    """
    Get input and target transforms for the Ini30Dataset.

    Parameters:
        num_bins (int): The number of bins used for transformation.
        augmentations (bool, optional): If True, apply data augmentations. Defaults to False.

    Returns:
        Tuple[Compose, FromPupilCenterToBoundingBox]: A tuple containing the input and target transforms.
    """
    sensor_size = (dataset_params["img_height"], dataset_params["img_width"], dataset_params["input_channel"])

    target_transforms = FromPupilCenterToBoundingBox(   yolo_loss=training_params["arch_name"][:6] == "retina",  
                                                        bbox_w=training_params["bbox_w"],
                                                        SxS_Grid=training_params["SxS_Grid"],
                                                        num_classes=training_params["num_classes"],
                                                        num_boxes=training_params["num_boxes"],
                                                        dataset_name=dataset_params["dataset_name"],
                                                        image_size=(dataset_params["img_width"], dataset_params["img_height"]),
                                                        num_bins=dataset_params["num_bins"]) 

    input_transforms = [AedatEventsToXYTP()]
    
    if dataset_params["dataset_name"]=="ini-30"and (dataset_params["img_width"] != 640 or dataset_params["img_height"] != 480):
        input_transforms.append(CenterCrop(sensor_size=(640, 480), size=(512, 512))) 
        input_transforms.append(Downscale())

    if dataset_params["pre_decimate"]:
        input_transforms.append(Decimation(dataset_params["pre_decimate_factor"]))
        
    if dataset_params["denoise_evs"]:
        input_transforms.append(Denoise(filter_time=dataset_params["filter_time"]))
    
    if dataset_params["random_flip"]:
        input_transforms.append(RandomFlipPolarity())

    if dataset_params["event_drop"]:
        input_transforms.append(EventDrop(sensor_size=sensor_size))
        
    if dataset_params["input_channel"] == 1:
        input_transforms.append(MergePolarities())
    
    input_transforms.append(ToFrame(sensor_size=sensor_size,   
                                    n_event_bins=dataset_params["num_bins"]))
    input_transforms = Compose(input_transforms)
    return input_transforms, target_transforms