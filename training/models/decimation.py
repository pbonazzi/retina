import torch
from torch import nn
import sinabs

class DecimationLayer(nn.Module):
    def __init__(
        self,
        spike_layer_class,
        batch_size: int,
        num_channels: int,
        decimation_rate: float
    ):
        super().__init__() 
        self.spike_layer_class = eval(spike_layer_class)  

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)
        
        self.spk = self.spike_layer_class(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=decimation_rate)
        
        # Prevent kernel from being trained
        self.conv.requires_grad_(False)
        self.conv.weight.data = (torch.eye(num_channels, num_channels).unsqueeze(-1).unsqueeze(-1) )

    def forward(self, x):

        # Conv expects 4D input
        input_dims = len(x.shape)
        if input_dims == 5:
            (n, t, c, h, w) = x.shape
            x = x.reshape((n * t, c, h, w))

        x = self.conv(x)  # (nt, c, h, w) 
        x = self.spk(x)  # (n, t, c, h, w)

        if input_dims == 5:
            # Bring to original shape
            x = x.reshape((n, t, c, h, w))
        return x