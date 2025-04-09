import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            stride=1, 
            padding=0, 
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
