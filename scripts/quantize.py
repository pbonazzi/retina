import tensorflow as tf
import numpy as np
import os, pdb
from torch.utils.data import RandomSampler 

from data.module import EyeTrackingDataModule 
from data.utils import load_yaml_config

# Path
path_to_run = "./output/experiment/"

# Load dataset params
training_params = load_yaml_config(os.path.join(path_to_run, "training_params.yaml"))
dataset_params = load_yaml_config(os.path.join(path_to_run, "dataset_params.yaml"))  

data_module = EyeTrackingDataModule(dataset_params=dataset_params, training_params=training_params, num_workers=16)
data_module.setup(stage='fit')
sampler = RandomSampler(data_module.train_dataset, replacement=True, num_samples=10)
train_dataloader = data_module.train_dataloader(sampler)

def representative_dataset():
    for input_data, _ in train_dataloader:  
        input_data = input_data.detach().cpu().numpy().astype(np.float32)  # Convert PyTorch tensor to NumPy
        print(f"Step {step}: Input shape = {input_data.shape}")  # Debugging: Check shape
        yield [input_data]  # Wrap in a list for TensorFlow compatibility

# Test
model = tf.saved_model.load(os.path.join(path_to_run,"models/model_tf"))
concrete_func = model.signatures["serving_default"]
input_shape = concrete_func.inputs[0].shape
print(f"Expected model input shape: {input_shape}") 

# Load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path_to_run,"models/model_tf"))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.float32 if mixed precision is allowed
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save the quantized model
with open(os.path.join(path_to_run, "models/model_int8.tflite"), "wb") as f:
    f.write(tflite_quant_model)

print("Fully quantized TFLite model saved as model_int8.tflite")
