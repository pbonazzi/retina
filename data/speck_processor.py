import torch, math, pdb
import numpy as np
from typing import List, Tuple

def events_to_label( 
    events: List, 
    shape: Tuple
) -> torch.Tensor:
    """
    Convert events from DynapcnnNetworks to labels raster 

    Parameters
    ----------

    events: List[Spike]
        A list of events that will be streamed to the device 
    shape: Optional[Tuple]
        Shape of the raster to be produced, (Time, Channel)

    Returns
    -------
    raster: torch.Tensor
        A 4 dimensional tensor of spike events with the dimensions [Time, Channel]
    """

    raster = torch.zeros(shape)

    if len(events) == 0:
        return raster
    if shape[0] == 1:
        for event in events:  
            raster[0, event.feature] += 1 
        return raster

    # Timestamps are in microseconds
    timestamps = [event.timestamp for event in events]
    start_timestamp, end_timestamp = min(timestamps), max(timestamps)
    if start_timestamp == end_timestamp: 
        for i, event in enumerate(events):  
            raster[i//shape[0], event.feature] += 1 
            
    dt = math.ceil((end_timestamp - start_timestamp)/shape[0])
    
    # dt in microseconds (same unit as event timestamps) 
    for event in events:  
        raster[int((event.timestamp - start_timestamp)/dt), event.feature] += 1 
    return raster

def events_to_raster(events: List, shape) -> torch.Tensor:
    """
    Convert events from DynapcnnNetworks to spike raster

    Parameters
    ----------

    events: List[Spike]
        A list of events that will be streamed to the device
    dt: float
        Length of each time step for rasterization
    shape: Optional[Tuple]
        Shape of the raster to be produced, excluding the time dimension. (Channel, Height, Width)
        If this is not specified, the shape is inferred based on the max values found in the events.

    Returns
    -------
    raster: torch.Tensor
        A 4 dimensional tensor of spike events with the dimensions [Time, Channel, Height, Width]
    """
    # Timestamps are in microseconds
    timestamps = [event.timestamp for event in events]
    start_timestamp, end_timestamp = min(timestamps), max(timestamps)
    dt = math.ceil((end_timestamp - start_timestamp)/shape[0])
    
    # Initialize an empty raster
    raster = torch.zeros(shape)


    for event in events:
        raster[
            int((event.timestamp - start_timestamp)/dt)-1,
            event.feature,
            event.x,
            event.y,
        ] += 1
    return raster

def label_to_bbox( 
    predictions: torch.tensor
) -> torch.Tensor:
    """dt

    Convert events from DynapcnnNetworks to labels raster 

    Parameters
    ----------

    events: List[Spike]
        A list of events that will be streamed to the device 
    shape: Optional[Tuple]
        Shape of the raster to be produced, (Time, Channel)

    Returns
    -------
    raster: torch.Tensor
        A 4 dimensional tensor of spike events with the dimensions [Time, Channel]
    """

    # Timestamps are in microseconds
    
    # dt in microseconds (same unit as event timestamps) 
    predictions = predictions.reshape(-1, 5, 5, 11)
        
    # Fix the bbox size
    norm_pred1 = norm_pred2 = torch.zeros_like(predictions[..., 2:6])

    point_1 = (predictions[..., 2:6] [..., :2] + (predictions[..., 2:6] [..., 2:] -predictions[..., 2:6] [..., :2])/2)
    point_2 = (predictions[..., 7:11][..., :2] + (predictions[..., 7:11][..., 2:] -predictions[..., 7:11][..., :2])/2)
    
    norm_pred1[...,:2] = point_1 - 5/64
    norm_pred1[...,2:] = point_1 + 5/64
    predictions[..., 2:6] = norm_pred1
    
    norm_pred2[...,:2] = point_2 - 5/64
    norm_pred2[...,2:] = point_2 + 5/64
    predictions[..., 7:11] = norm_pred2

    # Find the index of the maximum confidence score
    predictions = predictions.numpy()
    bbox_array, conf_array = [], []
    for pred in predictions:
        reshaped_predictions = pred.reshape((25, 11)) 
        conf_scores_1 = reshaped_predictions[:, 1:2]#*reshaped_predictions[:, 0:1]
        conf_scores_2 = reshaped_predictions[:, 6:7]#*reshaped_predictions[:, 0:1]

        best_bbox_index_1 = np.argmax(conf_scores_1)
        best_bbox_index_2 = np.argmax(conf_scores_2)

        # Compare 
        if conf_scores_1[best_bbox_index_1] >= conf_scores_2[best_bbox_index_2]:
            bbox_array.append(reshaped_predictions[best_bbox_index_1][2:6].clip(0,1)*64)
            conf_array.append(conf_scores_1[best_bbox_index_1])
        else:
            bbox_array.append(reshaped_predictions[best_bbox_index_2][7:11].clip(0,1)*64)
            conf_array.append(conf_scores_2[best_bbox_index_2])

    return np.stack(bbox_array), np.stack(conf_array)