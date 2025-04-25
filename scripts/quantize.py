import os
# ❌ Disable GPU BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import os, pdb
from torch.utils.data import RandomSampler 

from data.module import EyeTrackingDataModule 
from data.utils import load_yaml_config

# Path
path_to_run = "./output/retina-ann-test/"

# Load dataset params
training_params = load_yaml_config(os.path.join(path_to_run, "training_params.yaml"))
dataset_params = load_yaml_config(os.path.join(path_to_run, "dataset_params.yaml"))  

data_module = EyeTrackingDataModule(dataset_params=dataset_params, training_params=training_params, num_workers=16)
data_module.setup(stage='fit')
sampler = RandomSampler(data_module.train_dataset, replacement=True, num_samples=10)
train_dataloader = data_module.train_dataloader(sampler)

# Representative dataset function for calibration
def representative_dataset():
    for input_data, _ in train_dataloader:  
        input_data = input_data.detach().cpu().numpy().astype(np.float32)
        for i in range(input_data.shape[0]):  # Ensure per-sample yield
            yield [input_data[i:i+1]]

# Load the SavedModel
model_path = os.path.join(path_to_run, "models/model_tf_2")
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

# Set optimizations and quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
tflite_path = os.path.join(path_to_run, "models/model_int8_2.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print("✅ Fully quantized TFLite model saved as model_int8_2.tflite")

