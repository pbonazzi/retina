import torch
import torch.nn as nn
import sinabs
import pdb
try:
    import sinabs.exodus as ex
    exodus_installed = True
except:
    exodus_installed = False



def compute_output_dim(training_params):
    # select loss
    if training_params["arch_name"] == "3et" or (not training_params["use_yolo_loss"]):
        output_dim  = 2 
    else:
        output_dim  = training_params["SxS_Grid"]  * training_params["SxS_Grid"] \
            *(training_params["num_classes"] + training_params["num_boxes"] * 5) 
    return output_dim

def estimate_activations_size(x):
    """
    Recursively computes the total number of elements across all tensors in
    a nested structure (lists, tuples).
    """
    num_activations = 0

    if isinstance(x, torch.Tensor):
        # If it's a tensor, just add its numel
        num_activations += x.numel()
    elif isinstance(x, (tuple, list)):
        # If it's a tuple or list, iterate through its elements
        for item in x:
            num_activations += estimate_activations_size(item)
    
    return num_activations

def estimate_model_size(model, input_tensor):
    """
    Estimate the total activation size of the model for a given input tensor.
    
    Parameters:
    - model: PyTorch model (e.g., Baseline_3ET)
    - input_tensor: Tensor with shape (batch_size, time_steps, channels, height, width)
    
    Returns:
    - total_activations: Total number of activations in the model.
    """
    # modules
    from training.models.spiking.decimation import DecimationLayer
    from .baseline_3et import ConvLSTM
    from .binarization.binary_operator import DoReFaConv2d, DoReFaLinear

    # accumulate activatioms
    total_activations = 0
    
    # Define a hook function to calculate the activations at each layer
    def hook_fn(module, input, output):
        nonlocal total_activations
        # Calculate the number of activations for this layer
        total_activations += estimate_activations_size(output)  # numel returns the total number of elements in the tensor
    
    # Recursively register hooks for all submodules
    def register_hooks(module):
        if isinstance(module, (ConvLSTM, nn.BatchNorm3d, nn.MaxPool3d, nn.Dropout, nn.Linear, nn.Conv2d, DoReFaConv2d, DoReFaLinear, nn.BatchNorm2d)):
            module.register_forward_hook(hook_fn)
        for child in module.children():
            register_hooks(child)
    
    # Register hooks at all levels of the model
    register_hooks(model)
    
    # Perform a forward pass with the given input tensor to trigger the hooks
    with torch.no_grad():
        model(input_tensor)

    return total_activations


def convert_exodus_to_sinabs(snn_model):
    if exodus_installed:
        for i, layer in enumerate(snn_model.children()):
            if isinstance(layer, DecimationLayer):  
                snn_model[i].spk = sinabs.layers.IAFSqueeze(
                    batch_size=layer.spk.batch_size,
                    spike_threshold=layer.spk.spike_threshold,
                    min_v_mem=layer.spk.min_v_mem,
                    num_timesteps=layer.spk.num_timesteps,
                ) 
            elif isinstance(layer, ex.layers.IAFSqueeze):
                snn_model[i] = sinabs.layers.IAFSqueeze(
                    batch_size=layer.batch_size,
                    spike_threshold=layer.spike_threshold,
                    min_v_mem=layer.min_v_mem,
                    num_timesteps=layer.num_timesteps,
                ) 
    return snn_model

def convert_sinabs_to_exodus(snn_model):
    if exodus_installed:
        for i, layer in enumerate(snn_model.children()):
            if isinstance(layer, DecimationLayer):  
                snn_model[i].spk = ex.layers.IAFSqueeze(
                    batch_size=layer.spk.batch_size,
                    spike_threshold=layer.spk.spike_threshold,
                    min_v_mem=layer.spk.min_v_mem,
                    num_timesteps=layer.spk.num_timesteps,
                ) 
            elif isinstance(layer, sinabs.layers.IAFSqueeze):
                snn_model[i] = ex.layers.IAFSqueeze(
                    batch_size=layer.batch_size,
                    spike_threshold=layer.spike_threshold,
                    min_v_mem=layer.min_v_mem,
                    num_timesteps=layer.num_timesteps,
                ) 
    return snn_model

def convert_to_dynap(snn_model, input_shape, dvs_input=False):
    from sinabs.backend.dynapcnn import DynapcnnNetwork
 
    with torch.no_grad():
        dynapp_model = []

        for layer in snn_model.children():
            # Convert Decimation Layer
            if isinstance(layer, DecimationLayer):
                dynapp_model.append(layer.conv)
                dynapp_model.append(layer.spk)
            
            # Fuse BatchNorm2D
            elif isinstance(layer, torch.nn.BatchNorm2d):
                assert isinstance(dynapp_model[-1], torch.nn.Conv2d)
                dynapp_model[-1] = fuse_conv_bn(dynapp_model[-1], layer) 
            else:
                dynapp_model.append(layer)

        snn_model = torch.nn.Sequential(*dynapp_model)
    
    # DYNAP-CNN compatible network
    dynapcnn_net = DynapcnnNetwork(
        snn_model,
        input_shape=input_shape,
        discretize=True, 
        dvs_input=dvs_input,
    )

    return dynapcnn_net

def convert_to_n6(ann_model, input_shape, dvs_input=False): 
    
    ann_model.eval()
 
    with torch.no_grad():
        layers_model = []

        for layer in ann_model.children():  
            if isinstance(layer, torch.nn.BatchNorm2d):
                assert isinstance(layers_model[-1], torch.nn.Conv2d)
                layers_model[-1] = fuse_conv_bn(layers_model[-1], layer) 
            else:
                layers_model.append(layer)

        ann_model = torch.nn.Sequential(*layers_model)

    return ann_model

def get_spiking_threshold_list(snn_model):
    # compute spiking thresholds for input loss
    spiking_thresholds = [] 
    for layer in snn_model:
        if isinstance(layer, DecimationLayer):
            spiking_thresholds.append(layer.spk.spike_threshold.item())
    
        elif isinstance(layer, sinabs.layers.IAFSqueeze):
            spiking_thresholds.append(layer.spike_threshold.item())
            
        elif isinstance(layer, sinabs.layers.IAF):
            spiking_thresholds.append(layer.spike_threshold.item())
                        
        elif exodus_installed:
            if isinstance(layer, ex.layers.IAFSqueeze):
                spiking_thresholds.append(layer.spike_threshold.item())
            elif isinstance(layer, ex.layers.IAF):
                spiking_thresholds.append(layer.spike_threshold.item())

    return spiking_thresholds

def fuse_conv_bn(conv, bn):
    
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=False
    )
    
    # fuse weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    bn_std = torch.sqrt(bn.running_var + bn.eps)
    w_bn = torch.diag(bn.weight.div(bn_std))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))


    return fused
