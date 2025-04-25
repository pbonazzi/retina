from training.models.utils import compute_output_dim

def get_retina_model_configs(dataset_params, training_params, quant_params):
    
    # Spiking
    if training_params["arch_name"] =="retina_snn":
        if dataset_params["img_width"] == 64 or dataset_params["img_height"] == 64:
            layers_config = config_retina_snn_for_64x64(dataset_params, training_params)
        elif dataset_params["img_width"] == 128 or dataset_params["img_height"] == 128:
            layers_config = config_retina_snn_for_128x128(dataset_params, training_params)

    # Artificial
    elif training_params["arch_name"] =="retina_ann": 
        if dataset_params["img_width"] == 64 or dataset_params["img_height"] == 64:
            layers_config = config_retina_for_64x64(dataset_params, training_params, quant_params) 
        elif dataset_params["img_width"] == 128 or dataset_params["img_height"] == 128:
            layers_config = config_retina_for_128x128(dataset_params, training_params, quant_params) 

    return layers_config

# Spiking
def config_retina_snn_for_128x128(dataset_params, training_params):
    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                },
                {"name": "Decimation", "decimation_rate": dataset_params["decimation_rate"]},
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
                {"name": "Linear", "out_dim": compute_output_dim(training_params)},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
            ]

    return layers_config

def config_retina_snn_for_64x64(dataset_params, training_params):
    
    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                },
                {"name": "Decimation", "decimation_rate": dataset_params["decimation_rate"]},
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
                {"name": "Linear", "out_dim": compute_output_dim(training_params)},
                {"name": "IAF", "tau_syn": None, "spike_threshold": 1, "min_v_mem": -1.0},
            ]

    return layers_config



def config_retina_for_128x128(dataset_params, training_params, quant_params):
    
    if quant_params["a_bit"] == 1 or quant_params["w_bit"] == 1:
        conv2d_name = "DoReFaConv2d"
        linear_name = "DoReFaLinear"
    else:
        conv2d_name = "Conv"
        linear_name = "Linear"

    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                }, 
                # Layer 1
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 2
                {"name": conv2d_name, "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 16, "s_xy": 2},
                # Layer 3
                {"name": conv2d_name, "out_dim": 64, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                # Layer 4
                {"name": conv2d_name, "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 5
                {"name": conv2d_name, "out_dim": 128, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                # Layer 6
                {"name": conv2d_name, "out_dim": 256, "k_xy": 3, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                 {"name": "ReLu"},
                # Layer 7
                {"name": "Flat"},
                {"name": "Linear", "out_dim": 512},
                 {"name": "ReLu"},
                # Layer 8
                {"name": "Linear", "out_dim": compute_output_dim(training_params)},
                 {"name": "ReLu"},
            ]

    return layers_config

def config_retina_for_64x64_v1(dataset_params, training_params, quant_params):
    
    if quant_params["a_bit"] == 1 or quant_params["w_bit"] == 1:
        conv2d_name = "DoReFaConv2d"
        linear_name = "DoReFaLinear"
    else:
        conv2d_name = "Conv"
        linear_name = "Linear"

    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                }, 
                # Layer 1
                {"name": conv2d_name, "out_dim": 16, "k_xy": 5, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 2
                {"name": conv2d_name, "out_dim": 64, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 3
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 4
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                # Layer 5
                {"name": conv2d_name, "out_dim": 8, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"}, 
                # Layer 6
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                # Layer 7
                {"name": "Flat"},
                {"name": linear_name, "out_dim": 128},
                {"name": "ReLu"},
                # Layer 8
                {"name": linear_name, "out_dim": compute_output_dim(training_params)}
            ]

    return layers_config

def config_retina_for_64x64(dataset_params, training_params, quant_params):
    
    if quant_params["a_bit"] == 1 or quant_params["w_bit"] == 1:
        conv2d_name = "DoReFaConv2d"
        linear_name = "DoReFaLinear"
    else:
        conv2d_name = "Conv"
        linear_name = "Linear"

    layers_config = [
                # Layer 0
                {
                    "name": "Input",
                    "img_width": dataset_params["img_width"],
                    "img_height": dataset_params["img_height"],
                    "input_channel": dataset_params["input_channel"],
                }, 
                # Layer 1
                {"name": conv2d_name, "out_dim": 16, "k_xy": 5, "s_xy": 2, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                # Layer 2
                {"name": conv2d_name, "out_dim": 64, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                # Layer 3
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                # Layer 4
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 5
                {"name": conv2d_name, "out_dim": 8, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"}, 
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 6
                {"name": conv2d_name, "out_dim": 16, "k_xy": 3, "s_xy": 1, "p_xy": 1},
                {"name": "BatchNorm"},
                {"name": "ReLu"},
                {"name": "AvgPool", "k_xy": 2, "s_xy": 2},
                # Layer 7
                {"name": "Flat"},
                {"name": linear_name, "out_dim": 128},
                {"name": "ReLu"},
                # Layer 8
                {"name": linear_name, "out_dim": compute_output_dim(training_params)}
            ]

    return layers_config