import torch, pdb
from thop import profile

from training.models.baseline_3et import Baseline_3ET
from training.models.retina.retina import Retina
from training.models.retina.helper import get_retina_model_configs

from training.models.utils import estimate_model_size
from data.utils import load_yaml_config

# Input Tensor
params = load_yaml_config("configs/default.yaml")
training_params = params["training_params"] 
dataset_params = params["dataset_params"] 
quant_params = params["quant_params"] 
input_shape = (
    dataset_params["input_channel"],
    dataset_params["img_width"],
    dataset_params["img_height"]
) 
num_bins = dataset_params["num_bins"]
training_params["batch_size"] = 1

if training_params["arch_name"] == "3et":
    input_tensor = torch.randn(training_params["batch_size"], num_bins, *input_shape)
    model = Baseline_3ET(input_shape[2], input_shape[1], input_shape[0])
else:
    input_tensor = torch.randn(training_params["batch_size"]*num_bins, *input_shape)
    layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
    model = Retina(dataset_params, training_params, layers_config) 

# Calculate MACs
macs, params = profile(model, inputs=(input_tensor,))
print(f"Number of MAC operations: {macs}")

# Calculate the size of the activations
num_activations = estimate_model_size(model, input_tensor)
size_activations_bytes = num_activations * quant_params["a_bit"]/ 8
print(f"Number of Activations: {num_activations}")
size_activations_mb = size_activations_bytes / (1024**2)
print('Activations size: {:.1f}MB'.format(size_activations_mb))

# Calculate the size of the weights (model parameters)
num_parameters = sum(p.numel() for p in model.parameters())
size_weights_bytes = num_parameters * quant_params["w_bit"]/ 8
print(f"Number of Parameters: {num_parameters}") 
size_weights_mb = size_weights_bytes / (1024**2)
print('Weights size: {:.1f}MB'.format(size_weights_mb))

# Calculate the size of the buffer
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
buffer_size_bytes = buffer_size * quant_params["w_bit"]/ 8 

# Total model size (weights + activations)
model_size_bytes = size_weights_bytes + size_activations_bytes
model_size_mb = model_size_bytes / (1024**2)
print('Weights+activations size: {:.1f}MB'.format(model_size_mb))

# Total model size (weights + buffers)
model_size_bytes = size_weights_bytes + buffer_size_bytes
model_size_kb = (model_size_bytes / (1024**2))*1000
print('Weights+buffers size: {:.1f}Kb'.format(model_size_kb))