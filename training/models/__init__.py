import torch
import sinabs
import pdb
from training.models.decimation import DecimationLayer
try:
    import sinabs.exodus as ex
    exodus_installed = True
except:
    exodus_installed = False

def get_model_for_baseline(dataset_params, training_params):
    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                },
                {"name": "Decimation", "decimation_rate": training_params["decimation_rate"]},
                # Layer 1
                {"name": "Conv", "out_dim": 16, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 2, "s_xy": 2},
                # Layer 2
                {"name": "Conv", "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 16, "s_xy": 2},
                # Layer 3
                {"name": "Conv", "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 2, "s_xy": 2},
                # Layer 4
                {"name": "Conv", "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 5
                {"name": "Conv", "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 6
                {"name": "Conv", "out_dim": 256, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 7
                {"name": "Flat"},
                {"name": "Linear", "out_dim": 512},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 8
                {"name": "Linear", "out_dim": training_params["output_dim"]},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
            ]

    return layers_config

def get_model_for_speck(dataset_params, training_params):
    
    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                },
                {"name": "Decimation", "decimation_rate": training_params["decimation_rate"]},
                # Layer 1
                {"name": "Conv", "out_dim": 16, "k_xy": 5, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 2, "s_xy": 2},
                # Layer 2
                {"name": "Conv", "out_dim": 64, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 2, "s_xy": 2},
                # Layer 3
                {"name": "Conv", "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                {"name": "SumPool", "k_xy": 2, "s_xy": 2},
                # Layer 4
                {"name": "Conv", "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 5
                {"name": "Conv", "out_dim": 8, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 6
                {"name": "Conv", "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 7
                {"name": "Flat"},
                {"name": "Linear", "out_dim": 128},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
                # Layer 8
                {"name": "Linear", "out_dim": training_params["output_dim"]},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
            ]

    return layers_config


def compute_output_dim(training_params):
    # select loss
    if training_params["yolo_loss"]:
        output_dim  = training_params["SxS_Grid"]  * training_params["SxS_Grid"] \
            *(training_params["num_classes"] + training_params["num_boxes"] * 5)
    elif training_params["focal_loss"]:
        output_dim = 4
    else:
        output_dim  = 2
    return output_dim

def get_summary(model):
    """
    Prints model memory
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_kb = ((param_size + buffer_size) / 1024**2)*1000
    print('Model size: {:.1f}KB'.format(size_all_kb))

    # Count the number of layers
    num_layers = sum(1 for _ in model.modules()) - 1
    print(f"Number of layers: {num_layers}")

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}") 

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
    
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False
    )

    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fused.weight.copy_( torch.mm(w_bn, w_conv).view(fused.weight.size()) )


    return fused
