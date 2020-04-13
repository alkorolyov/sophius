import torch
import torch.nn as nn

class _ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self._shortcut(x) + self._block(x)
        out = self.relu(out)
        return out

class ResNetBlock(_ResNetBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
        conv2 = nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3, 
                          stride=1,
                          padding=1,
                          bias=False)
        bn1 = nn.BatchNorm2d(num_features=out_channels)
        bn2 = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU(inplace=True)
        self._block = nn.Sequential(conv1, bn1, relu,
                                    conv2, bn2)
        self._shortcut = nn.Identity()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


class ResNetSkipBlock(_ResNetBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        conv2 = nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
        bn1 = nn.BatchNorm2d(num_features=out_channels)
        bn2 = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU(inplace=True)
        self._block = nn.Sequential(conv1, bn1, relu,
                                    conv2, bn2)
        conv_shortcut = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1, 
                                  stride=2,
                                  padding=0,
                                  bias=False)
        bn3 = nn.BatchNorm2d(num_features=out_channels)
        self._shortcut = nn.Sequential(conv_shortcut, bn3)