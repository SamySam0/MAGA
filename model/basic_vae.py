import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_res_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_res_hiddens,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_res_hiddens,
                out_channels=num_hiddens,
                kernel_size=1, stride=1, bias=False,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_res_layers, num_res_hiddens):
        super().__init__()
        assert in_channels == num_hiddens, 'Input and output channel of each residual layer must be of the same size!'
        self.num_res_layers = num_res_layers
        self.layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_res_hiddens)
            for _ in range(self.num_res_layers)
        ])
    
    def forward(self, x):
        for i in range(self.num_res_layers):
            x = self.layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_res_layers, num_res_hiddens):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens//2,
            kernel_size=4, stride=2, padding=1,
        ) # Downscale HxW by two
        self.conv_2 = nn.Conv2d(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=4, stride=2, padding=1,
        ) # Downscale initial HxW by four
        self.conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3, stride=1, padding=1,
        ) # Linear scaling

        self.residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_res_layers=num_res_layers,
            num_res_hiddens=num_res_hiddens,
        ) # Linear residual
    
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.conv_3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_res_layers, num_res_hiddens):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3, stride=1, padding=1,
        ) # Linear scale, changes channel depth

        self.residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_res_layers=num_res_layers,
            num_res_hiddens=num_res_hiddens,
        ) # Linear residual

        self.conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=4, stride=2, padding=1,
        ) # ConvTranspose upsamples the size by two

        self.conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens//2,
            out_channels=out_channels,
            kernel_size=4, stride=2, padding=1,
        ) # Upsample size by two and recover input's channel depth
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans_1(x))
        return self.conv_trans_2(x)
