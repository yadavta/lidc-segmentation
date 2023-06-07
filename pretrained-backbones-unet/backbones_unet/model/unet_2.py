import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from functools import reduce
from operator import __add__

class Unet2(nn.Module):
    def __init__(
            self,
            in_channels=1,
            num_classes=1,
    ):
        super().__init__()
        # encoder 
        self.conv1 = Conv2dBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = Conv2dBlock(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = Conv2dBlock(64, 128)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4 = Conv2dBlock(128, 256)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.conv5 = Conv2dBlock(256, 512)
        
        # decoder
        self.convt6 = Conv2dTranspose(512, 256)
        self.conv6 = Conv2dBlock(512, 256)
        self.convt7 = Conv2dTranspose(256, 128)
        self.conv7 = Conv2dBlock(256, 128)
        self.convt8 = Conv2dTranspose(128, 64)
        self.conv8 = Conv2dBlock(128, 64)
        self.convt9 = Conv2dTranspose(64, 32)
        self.conv9 = Conv2dBlock(64, 32)
        self.conv10 = nn.Conv2d(32, 1, (1, 1), bias=True)

    def forward(self, x: torch.Tensor):
        # encode
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        
        # decode
        u6 = torch.cat((self.convt6(c5), c4), dim=1)
        c6 = self.conv6(u6)
        u7 = torch.cat((self.convt7(c6), c3), dim=1)
        c7 = self.conv7(u7)
        u8 = torch.cat((self.convt8(c7), c2), dim=1)
        c8 = self.conv8(u8)
        u9 = torch.cat((self.convt9(c8), c1), dim=1)
        c9 = self.conv9(u9)
        c10 = self.conv10(c9)
        
        return c10

    @torch.no_grad()
    def predict(self, x):
        """
        Inference method. Switch model to `eval` mode, 
        call `.forward(x)` with `torch.no_grad()`
        Parameters
        ----------
        x: torch.Tensor
            4D torch tensor with shape (batch_size, channels, height, width)
        Returns
        ------- 
        prediction: torch.Tensor
            4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training: self.eval()
        x = self.forward(x)
        return x

# UNet Blocks

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding='same', stride=1, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=True)
        if norm_layer is not None:
            self.bn = norm_layer(out_channels)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
         
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Conv2dTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), output_padding=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x = self.up(x)
        return x
    