import torch, pdb
import torch.nn as nn
import sinabs
import sinabs.activation as sina

from training.models.utils import get_summary
from training.models.blocks.decimation import DecimationLayer


class Retina(nn.Module):
    def __init__(self, dataset_params, training_params, layers_config):
        super(Retina, self).__init__()

        self.train_with_mem = training_params["train_with_mem"]
        self.train_with_exodus = training_params["train_with_exodus"]

        # data configs
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # train configs
        self.training_params = training_params

        # spiking layer activations
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

                # weights init
                torch.nn.init.xavier_uniform_(modules[-1].weight)

            elif layer["name"] == "ReLu":
                modules.append(nn.ReLU())

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

            elif layer["name"] == "Flat":
                modules.append(nn.Flatten())

                # out dimensions
                c_in = c_x * c_y * c_in

            elif layer["name"] == "Linear":
                # input dimensions
                print(layer["name"], " in: \n ", c_in)
                modules.append(nn.Linear(c_in, layer["out_dim"], bias=False))

                # out dimensions
                print(layer["name"], " out: \n ", layer["out_dim"])
                c_in = layer["out_dim"]

                # weights init
                torch.nn.init.xavier_uniform_(modules[-1].weight)

            elif layer["name"] == "SumPool":
                # input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(sinabs.layers.SumPool2d(layer["k_xy"], layer["s_xy"]))

                # out dimensions
                c_x, c_y = (c_x - layer["k_xy"]) // layer["s_xy"] + 1, (
                    c_y - layer["k_xy"]
                ) // layer["s_xy"] + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)

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
                        record_states=self.train_with_mem,
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
        # print("Number of MAC", self.compute_mac_operations())
        get_summary(self)

    def compute_mac_operations(self):
        total_mac_ops = 0
        input_size = (
            self.training_params["batch_size"] * self.num_bins,
            self.input_channel,
            self.img_width,
            self.img_height,
        )
        with torch.no_grad():
            x = torch.zeros(*input_size)
            for module in self.seq:
                x = module(x)
                if isinstance(module, nn.Conv2d):
                    total_mac_ops += (
                        module.in_channels
                        * module.out_channels
                        * module.kernel_size[0] ** 2
                        * x.size(-1)
                        * x.size(-2)
                    )
                elif isinstance(module, nn.Linear):
                    total_mac_ops += module.in_features * module.out_features

        return total_mac_ops

    def forward(self, x):
        return self.seq(x)
