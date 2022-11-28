# Load functions
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, ConvTranspose2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax, pad
from torch.nn import ReLU, Sigmoid, Softmax
import numpy as np
from torchvision.transforms import CenterCrop



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_kernel_stride=1, conv_padding=0):
        super(Conv, self).__init__()

        self.conv_1 = Conv2d(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=conv_kernel_size,
                            stride=conv_kernel_stride,
                            padding=conv_padding,
                            bias=False)

        self.conv_2 = Conv2d(in_channels=out_channels, 
                            out_channels=out_channels,
                            kernel_size=conv_kernel_size,
                            stride=conv_kernel_stride,
                            padding=conv_padding,
                            bias = False)

    def forward(self, x_img, activationFun="Relu"):
        if activationFun=="Relu":
            self.activation = ReLU()
        elif activationFun == "Sigmoid":
            self.activation = Sigmoid()
        else:
            raise ValueError('Invalid activation function')

        x_img = self.activation(self.conv_1(x_img))
        x_img = self.activation(self.conv_2(x_img))

        return x_img


class down_conv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_kernel_stride=1, conv_padding=0, maxpool_kernel_size = 2, maxpool_kernel_stride = 2):
        super(down_conv, self).__init__()

        self.maxpool = MaxPool2d(kernel_size = maxpool_kernel_size, stride = maxpool_kernel_stride)
        self.conv = Conv(in_channels, out_channels, conv_kernel_size, conv_kernel_stride, conv_padding)

    def forward(self, x_img):
        x_img = self.maxpool(x_img)
        x_img = self.conv(x_img)

        return x_img

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_kernel_stride=1, conv_padding=0, upconv_kernel_size=2, upconv_kernel_stride=1, upconv_padding=0):
        super(up_conv, self).__init__()

        self.upconv = ConvTranspose2d(in_channels = in_channels, 
                                        out_channels = out_channels, 
                                        kernel_size = upconv_kernel_size, 
                                        stride = upconv_kernel_stride, 
                                        padding = upconv_padding)

        self.conv = Conv(in_channels, out_channels, conv_kernel_size, conv_kernel_stride, conv_padding)


    def forward(self, x_img1, x_img2):
        x_img1 = self.upconv(x_img1)

        #x_img2 = CenterCrop(size = x_img1.size(dim=2))
        #x_img = torch.cat((x_img1, x_img2), dim = 1)  

        # Pad the upsampled image to fit the output from the skip connection
        diff = (x_img2.size(dim=2) - x_img1.size(dim=2))
        x_img1 = pad(x_img1, [diff//2, diff - diff//2, diff//2, diff - diff//2])

        x_img = torch.cat((x_img2, x_img1), dim = 1)  
        x_img = self.conv(x_img)

        return x_img


class final_conv(nn.Module):
    def __init__(self,in_channels, out_channels, conv_kernel_size=1, conv_kernel_stride=1, conv_padding=0):
        super(final_conv, self).__init__()

        self.conv = Conv2d(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=conv_kernel_size,
                            stride=conv_kernel_stride,
                            padding=conv_padding)
    
    def forward(self, x_img):
        x_img = self.conv(x_img)

        return x_img


class Unet(nn.Module):
    def __init__(self, im_height, im_width, im_channels, num_features, activationFun="Relu", maxpool_kernel_size=2, 
                maxpool_kernel_stride=2, conv_kernel_size = 3, conv_kernel_stride=1, conv_padding=0, 
                upconv_kernel_size=2, upconv_kernel_stride=1, upconv_padding=0):
        super(Unet, self).__init__()

        self.first_conv = Conv(in_channels=im_channels, out_channels=64,conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding)
        self.down_conv1 = down_conv(in_channels=64, out_channels=128,conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, maxpool_kernel_size = maxpool_kernel_size, maxpool_kernel_stride = maxpool_kernel_stride) 
        self.down_conv2 = down_conv(in_channels=128, out_channels=256,conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, maxpool_kernel_size = maxpool_kernel_size, maxpool_kernel_stride = maxpool_kernel_stride) 
        self.down_conv3 = down_conv(in_channels=256, out_channels=512,conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, maxpool_kernel_size = maxpool_kernel_size, maxpool_kernel_stride = maxpool_kernel_stride) 
        self.down_conv4 = down_conv(in_channels=512, out_channels=1024,conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, maxpool_kernel_size = maxpool_kernel_size, maxpool_kernel_stride = maxpool_kernel_stride) 
        self.up_conv1 = up_conv(in_channels=1024, out_channels=512, conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, upconv_kernel_size=upconv_kernel_size, upconv_kernel_stride=upconv_kernel_stride, upconv_padding=upconv_padding)
        self.up_conv2 = up_conv(in_channels=512, out_channels=256, conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, upconv_kernel_size=upconv_kernel_size, upconv_kernel_stride=upconv_kernel_stride, upconv_padding=upconv_padding)
        self.up_conv3 = up_conv(in_channels=256, out_channels=128, conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, upconv_kernel_size=upconv_kernel_size, upconv_kernel_stride=upconv_kernel_stride, upconv_padding=upconv_padding)
        self.up_conv4 = up_conv(in_channels=128, out_channels=64, conv_kernel_size=conv_kernel_size, conv_kernel_stride=conv_kernel_stride, conv_padding=conv_padding, upconv_kernel_size=upconv_kernel_size, upconv_kernel_stride=upconv_kernel_stride, upconv_padding=upconv_padding)
        self.conv_final = final_conv(in_channels=64, out_channels=num_features)

    
    def forward(self, x_img):
        out_first = self.first_conv(x_img)
        out_1 = self.down_conv1(out_first)
        out_2 = self.down_conv2(out_1)
        out_3 = self.down_conv3(out_2)
        out_4 = self.down_conv3(out_3)

        out = self.up_conv1(out_4, out_3)
        out = self.up_conv2(out, out_2)
        out = self.up_conv2(out, out_1)
        out = self.up_conv3(out, out_first)

        out = self.conv_final(out)

        return out

