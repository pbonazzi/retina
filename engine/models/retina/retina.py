import torch, pdb
import torch.nn as nn

import sinabs
import sinabs.activation as sina

from ..spiking.decimation import DecimationLayer
from ..binarization.binary_operator import DoReFaConv2d, DoReFaLinear 

from data.utils import load_yaml_config

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        ) 

    def forward(self, x):
        boxes = self.box_head(x) 
        return boxes
        
class Retina(nn.Module):
    def __init__(self, dataset_params, training_params, layers_config):
        super(Retina, self).__init__()

        # data configs
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # train configs
        self.training_params = training_params

        # spiking layer activations
        if self.training_params["arch_name"] =="retina_snn":
            spike_fn = (
                sina.MultiSpike if training_params["spike_multi"] else sina.SingleSpike
            )
            spike_reset = (
                sina.MembraneReset()
                if training_params["spike_reset"]
                else sina.MembraneSubtract()
            )
            spike_grad = sina.Heaviside(window=training_params["spike_window"])
            if training_params["spike_surrogate"]:
                spike_grad = (
                    sina.PeriodicExponential()
                    if training_params["spike_multi"]
                    else sina.SingleExponential()
                )

        # modules initialization
        modules = []
        for i, layer in enumerate(layers_config):
            if layer["name"] == "Input":
                c_x, c_y, c_in = (
                    layer["img_width"],
                    layer["img_height"],
                    layer["input_channel"],
                )
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in) 

            elif layer["name"] == "Conv":
                # input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=layer["out_dim"],
                        kernel_size=(layer["k_xy"], layer["k_xy"]),
                        stride=(layer["s_xy"], layer["s_xy"]),
                        padding=(layer["p_xy"], layer["p_xy"]),
                        bias=False,
                    )
                )

                # out dimensions
                c_in = layer["out_dim"]
                c_x = ((c_x - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                c_y = ((c_y - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in) 

            elif layer["name"] == "DoReFaConv2d":
                # input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(
                    DoReFaConv2d(
                        in_channels=c_in,
                        out_channels=layer["out_dim"],
                        kernel_size=(layer["k_xy"], layer["k_xy"]),
                        stride=(layer["s_xy"], layer["s_xy"]),
                        padding=(layer["p_xy"], layer["p_xy"]),
                        bias=False,
                    )
                )

                # out dimensions
                c_in = layer["out_dim"]
                c_x = ((c_x - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                c_y = ((c_y - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in) 

            elif layer["name"] == "DoReFaLinear":
                # input dimensions
                print(layer["name"], " in: \n ", c_in)
                modules.append(DoReFaLinear(in_features=c_in, out_features=layer["out_dim"], bias=False))

                # out dimensions
                print(layer["name"], " out: \n ", layer["out_dim"])
                c_in = layer["out_dim"] 

            elif layer["name"] == "Linear":
                # input dimensions
                print(layer["name"], " in: \n ", c_in)
                modules.append(nn.Linear(c_in, layer["out_dim"], bias=False))

                # out dimensions
                print(layer["name"], " out: \n ", layer["out_dim"])
                c_in = layer["out_dim"]

            elif layer["name"] == "PredictionHead":
                # input dimensions
                print(layer["name"], " in: \n ", c_in)
                modules.append(PredictionHead(c_in))

            elif layer["name"] == "ReLu":
                modules.append(nn.ReLU())

            elif layer["name"] == "Sigmoid":
                modules.append(nn.Sigmoid()) 

            elif layer["name"] == "BatchNorm":
                modules.append(nn.BatchNorm2d(c_in))

            elif layer["name"] == "AvgPool":
                # input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(nn.AvgPool2d(layer["k_xy"], layer["s_xy"]))

                # out dimensions
                c_x, c_y = (c_x - layer["k_xy"]) // layer["s_xy"] + 1, (
                    c_y - layer["k_xy"]
                ) // layer["s_xy"] + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)

            elif layer["name"] == "SumPool":
                # input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(sinabs.layers.SumPool2d(layer["k_xy"], layer["s_xy"]))

                # out dimensions
                c_x, c_y = (c_x - layer["k_xy"]) // layer["s_xy"] + 1, (
                    c_y - layer["k_xy"]
                ) // layer["s_xy"] + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)
                
            elif layer["name"] == "Flat":
                modules.append(nn.Flatten())

                # out dimensions
                c_in = c_x * c_y * c_in
                
            elif layer["name"] == "IAF":
                modules.append(
                    sinabs.layers.IAFSqueeze(
                        batch_size=training_params["batch_size"],
                        spike_fn=spike_fn,
                        surrogate_grad_fn=spike_grad,
                        reset_fn=spike_reset,
                        num_timesteps=self.num_bins,
                        tau_syn=layer["tau_syn"],
                        spike_threshold=layer["spike_threshold"],
                        record_states=True,
                        min_v_mem=layer["min_v_mem"],
                    )
                )

            elif layer["name"] == "Decimation":
                modules.append(
                    DecimationLayer(
                        spike_layer_class="sinabs.layers.iaf.IAFSqueeze",
                        decimation_rate=layer["decimation_rate"],
                        batch_size=training_params["batch_size"],
                        num_channels=self.input_channel,
                    )
                )
            else:
                raise NotImplementedError("Unknown Layer")

        self.seq = nn.Sequential(*modules) 

    def forward(self, x): 
        return self.seq(x)



if __name__ == "__main__":
    import torch 
    from .helper import get_retina_model_configs
    


    params = load_yaml_config("configs/default.yaml")
    training_params = params["training_params"]
    training_params["batch_size"] = 1
    dataset_params = params["dataset_params"] 
    quant_params = params["quant_params"] 

    layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
    model = Retina(dataset_params, training_params, layers_config) 

    input_shape = ( 
        dataset_params["input_channel"],
        dataset_params["img_width"],
        dataset_params["img_height"],
    )

    print(f"Input shape: {input_shape}")
    
    def prepare_input(resolution):
        t_b = dataset_params["num_bins"] * training_params["batch_size"]
        x1 = torch.FloatTensor(t_b, *resolution)  
        return dict(x = x1)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, input_shape, input_constructor=prepare_input)
    print(f"\n[ptflops] Model MACs: {macs}, Params: {params}\n")   

    from fvcore.nn import FlopCountAnalysis, parameter_count
    flops = FlopCountAnalysis(model, torch.randn(1, *input_shape))
    print(f"\n[fvcore] Model MACs: {flops.total() / 2}, Params: {sum(parameter_count(model).values())}\n")   

    from thop import profile
    macs, params = profile(model, inputs=(torch.randn(1, *input_shape),), verbose=True)
    print(f"\n[thop] MACs: {macs/1e6:.2f} M, Params: {params/1e3:.2f} K") 
