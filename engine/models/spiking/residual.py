from torch import nn
from sinabs.layers import IAFSqueeze


class ResidualBlock(nn.Module):

    def __init__(self):

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False)
        self.iaf1 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1), bias=False)
        self.iaf2 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

        self.conv3 = nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False)
        self.iaf3 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

    def forward(self, x):

        tmp = self.conv1(x)
        tmp = self.iaf1(tmp)
        out = self.conv2(tmp)
        out = self.iaf2(tmp)
        # residual connection
        out += tmp
        out = self.conv3(out)
        out = self.iaf3(out)

        return out
