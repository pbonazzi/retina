import yaml

import os
import torch
import pdb
import numpy as np
import onnx
import onnxruntime
from onnx import version_converter

def extract_xy_point_from_output(output, bbox_w, img_width):
    output = output.clone().detach()  

    def square_results(pred_boxes):
        norm_pred = torch.zeros_like(pred_boxes)
        center = pred_boxes[..., :2] + (pred_boxes[..., 2:] - pred_boxes[..., :2]) / 2
        norm_pred[..., :2] = center - bbox_w / img_width
        norm_pred[..., 2:] = center + bbox_w / img_width
        return norm_pred

    total_box = square_results(output) 
    center_point = total_box[:2] + (total_box[2:] - total_box[:2]) / 2 
    return center_point * img_width

def collect_statistics(distances):
    """
    Collect various statistics of a list of distances (e.g., float32_distances).
    
    Args:
        distances (list or np.array): List or numpy array of Euclidean distances.
    
    Returns:
        dict: Dictionary with statistics like mean, median, std, min, max, etc.
    """ 
    
    # Compute various statistics
    stats = {
        "mean": np.mean(distances),
        "median": np.median(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
        "25th_percentile": np.percentile(distances, 25),
        "75th_percentile": np.percentile(distances, 75),
        "iqr": np.percentile(distances, 75) - np.percentile(distances, 25),
        "count": len(distances)
    }
    return stats

def save_sample_batches(train_dataloader, save_path):
    
    # Set the save path 
    num_batches_to_save = 5
    saved_batches = []

    # Save batches from DataLoader
    for i, (input_data, target, _) in enumerate(train_dataloader):
        if i >= num_batches_to_save:
            break
        # Detach and convert to numpy
        np_input = input_data.detach().cpu().numpy()
        saved_batches.append(np_input)

    # Write to C header
    with open(save_path, "w") as f:
        f.write("#ifndef SAMPLE_BATCHES_H\n")
        f.write("#define SAMPLE_BATCHES_H\n\n")

        for batch_idx, batch in enumerate(saved_batches):
            batch = batch.astype(np.float32)
            flat_data = batch.flatten()
            shape = batch.shape  # (B, C, H, W)

            f.write(f"// Batch {batch_idx}, shape: {shape}\n")
            f.write(f"static const float sample_batch_{batch_idx}[] = {{\n")
            for i, value in enumerate(flat_data):
                f.write(f"{value:.6f}f")
                if i < len(flat_data) - 1:
                    f.write(", ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n")

        f.write("#endif // SAMPLE_BATCHES_H\n")
    

def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def change_opset(input_model: str, new_opset: int) -> str:
    """
    Converts the opset version of an ONNX model to a new opset version.

    Args:
        input_model (str): The path to the input ONNX model.
        new_opset (int): The new opset version to convert the model to.

    Returns:
        str: The path to the converted ONNX model.
    """
    if not input_model.endswith('.onnx'):
        raise Exception("Error! The model must be in onnx format")    
    model = onnx.load(input_model)
    # Check the current opset version
    current_opset = model.opset_import[0].version
    if current_opset == new_opset:
        print(f"The model is already using opset {new_opset}")
        return input_model

    # Modify the opset version in the model
    converted_model = version_converter.convert_version(model, new_opset)
    temp_model_path = input_model+ '.temp'
    onnx.save(converted_model, temp_model_path)

    # Load the modified model using ONNX Runtime Check if the model is valid
    session = onnxruntime.InferenceSession(temp_model_path)
    try:
        session.get_inputs()
    except Exception as e:
        print(f"An error occurred while loading the modified model: {e}")
        return

    # Replace the original model file with the modified model
    os.replace(temp_model_path, input_model)
    print(f"The model has been converted to opset {new_opset} and saved at the same location.")
    return input_model