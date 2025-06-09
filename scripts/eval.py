import onnxruntime as ort
import torch
import numpy as np
import pdb
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom imports
from data.utils import load_yaml_config, collect_statistics
from data.datasets.ini_30.helper import get_ini_30_dataset 

def evaluate_onnx_model(onnx_model_path, dataloader, dataset_params, device="cpu"):
    print(f"Loading ONNX model from {onnx_model_path}...")
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], sess_options=ort.SessionOptions())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    session.get_session_options().intra_op_num_threads = 4

    all_distances = []

    print("Starting evaluation...")
    for data, target, avg_dt, exp_id in tqdm(dataloader): 
        data = data.to(device).numpy().reshape(data.shape[0], -1, 64, 64) 
        preds = session.run([output_name], {input_name: data})[0]
        
        # Compute center of bounding box  
        preds_center_x = ((preds[:, 0] + preds[:, 2]) / 2) * dataset_params["img_width"]
        preds_center_y = ((preds[:, 1] + preds[:, 3]) / 2) * dataset_params["img_height"]
        preds_center = np.stack((preds_center_x, preds_center_y), axis=1)
        
        # Compute target center of bounding box
        target_center_x = ((target[:, 0] + target[:, 2]) / 2) * dataset_params["img_width"]
        target_center_y = ((target[:, 1] + target[:, 3]) / 2) * dataset_params["img_height"]
        target_center = np.stack((target_center_x, target_center_y), axis=1)
        
        # Compute Euclidean distances for each instance in the batch
        distances = np.linalg.norm(preds_center - target_center, axis=1)
        all_distances.extend(distances)
    
    return all_distances



if __name__ == "__main__":
    # Load configuration
    config = load_yaml_config("configs/default.yaml")
    training_params = config["training_params"] 
    dataset_params = config["dataset_params"]

    # Prepare validation dataset and dataloader
    val_dataset = get_ini_30_dataset("val", training_params, dataset_params)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate ONNX model
    onnx_model_path = "./output/model_float32.onnx"
    float32_distances = evaluate_onnx_model(
        onnx_model_path, val_loader, dataset_params
    )
    print(f"Float32 [all_stats] : {collect_statistics(float32_distances)}")

    # 8-bit Quantization
    onnx_model_path = "./output/model_int8.onnx"
    int8_distances = evaluate_onnx_model(
        onnx_model_path, val_loader, dataset_params
    )
    print(f"Int8 [all_stats] : {collect_statistics(int8_distances)}") 
    
    # Define academic-friendly figure size
    plt.figure(figsize=(6, 4))
    plt.hist(float32_distances, bins=30, color='#4682B4', alpha=0.7, label='Float32', edgecolor='black', linewidth=0.5) 
    plt.hist(int8_distances, bins=30, color='#FF7F50', alpha=0.7, label='Int8', edgecolor='black', linewidth=0.5) 
    plt.title('Euclidean Distance Distribution', fontsize=14)
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12) 
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend(fontsize=10) 
    plt.tight_layout()
    plt.savefig("figure.pdf") 