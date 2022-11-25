# Load functions
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, ConvTranspose2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax, one_hot
from torch.nn import ReLU, Sigmoid, Softmax
from torchvision.transforms import CenterCrop


class Unet_pp(nn.Module):
    def __init__(self, im_height, im_width, im_channels, num_features, activationFun="Relu",
                conv_kernel_size = 3, conv_kernel_stride=1, conv_padding=1, conv_dilation=1,
                conv_out_channels_0 = 32, upconv_kernel_size=3, upconv_kernel_stride=1, upconv_padding=1):
        super().__init__()

        if activationFun=="Relu":
            self.activation = ReLU()
        elif activationFun == "Sigmoid":
            self.activation = Sigmoid()
        else:
            raise ValueError('Invalid activation function')

        self.conv_00 = Conv2d(in_channels = im_channels,
                                out_channels = conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_10 = Conv2d(in_channels = conv_out_channels_0,
                                out_channels = 2*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_20 = Conv2d(in_channels = 2*conv_out_channels_0,
                                out_channels = 4*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_30 = Conv2d(in_channels = 4*conv_out_channels_0,
                                out_channels = 8*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_40 = Conv2d(in_channels = 8*conv_out_channels_0,
                                out_channels = 16*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)


        # Convolutional layers in skip-pathways
        self.conv_01 = Conv2d(in_channels = conv_out_channels_0 + conv_out_channels_0,
                                out_channels = conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_11 = Conv2d(in_channels = 2*conv_out_channels_0 + 2*conv_out_channels_0,
                                out_channels = 2*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_21 = Conv2d(in_channels = 4*conv_out_channels_0 + 4*conv_out_channels_0,
                                out_channels = 4*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_31 = Conv2d(in_channels = 8*conv_out_channels_0 + 8*conv_out_channels_0,
                                out_channels = 8*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)


        self.conv_02 = Conv2d(in_channels = conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0,
                                out_channels = conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_12 = Conv2d(in_channels = 2*conv_out_channels_0 + 2*conv_out_channels_0 + 2*conv_out_channels_0,
                                out_channels = 2*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_22 = Conv2d(in_channels = 4*conv_out_channels_0 + 4*conv_out_channels_0 + 4*conv_out_channels_0,
                                out_channels = 4*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_03 = Conv2d(in_channels = conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0,
                                out_channels = conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_13 = Conv2d(in_channels = 2*conv_out_channels_0 + 2*conv_out_channels_0 + 2*conv_out_channels_0 + 2*conv_out_channels_0,
                                out_channels = 2*conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_04 = Conv2d(in_channels = conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0 + conv_out_channels_0,
                                out_channels = conv_out_channels_0,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)

        self.conv_final = Conv2d(in_channels = conv_out_channels_0,
                                out_channels = num_features,
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)


        # Up-convolutional layers
        self.upconv_01 = ConvTranspose2d(in_channels = 2*conv_out_channels_0,
                                            out_channels = conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_11 = ConvTranspose2d(in_channels = 4*conv_out_channels_0,
                                            out_channels = 2*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_21 = ConvTranspose2d(in_channels = 8*conv_out_channels_0,
                                            out_channels = 4*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_31 = ConvTranspose2d(in_channels = 16*conv_out_channels_0,
                                            out_channels = 8*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)


        self.upconv_02 = ConvTranspose2d(in_channels = 2*conv_out_channels_0,
                                            out_channels = conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_12 = ConvTranspose2d(in_channels = 4*conv_out_channels_0,
                                            out_channels = 2*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_22 = ConvTranspose2d(in_channels = 8*conv_out_channels_0,
                                            out_channels = 4*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)


        self.upconv_03 = ConvTranspose2d(in_channels = 2*conv_out_channels_0,
                                            out_channels = conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.upconv_13 = ConvTranspose2d(in_channels = 4*conv_out_channels_0,
                                            out_channels = 2*conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)


        self.upconv_04 = ConvTranspose2d(in_channels = 2*conv_out_channels_0,
                                            out_channels = conv_out_channels_0,
                                            kernel_size = upconv_kernel_size,
                                            stride = upconv_kernel_stride,
                                            padding = upconv_padding)

        self.softmax = Softmax(dim=1)

    def forward(self, x_img):
        out = {}

        # Backbone
        out_conv_00 = self.conv_00(x_img)
        out_conv_00 = self.activation(out_conv_00)

        out_conv_10 = self.conv_10(out_conv_00)
        out_conv_10 = self.activation(out_conv_10)

        out_conv_20 = self.conv_20(out_conv_10)
        out_conv_20 = self.activation(out_conv_20)

        out_conv_30 = self.conv_30(out_conv_20)
        out_conv_30 = self.activation(out_conv_30)

        out_conv_40 = self.conv_40(out_conv_30)
        out_conv_40 = self.activation(out_conv_40)

        # First layer of skip-connections
        out_upconv_01 = self.upconv_01(out_conv_10)
        crop_00 = CenterCrop(size = out_upconv_01.size(dim=2))
        out_conv_00_crop = crop_00(out_conv_00)
        out_concat_01 = torch.cat((out_conv_00_crop, out_upconv_01), dim = 1)
        out_conv_01 = self.conv_01(out_concat_01)
        out_conv_01 = self.activation(out_conv_01)

        out_upconv_11 = self.upconv_11(out_conv_20)
        crop_10 = CenterCrop(size = out_upconv_11.size(dim=2))
        out_conv_10_crop = crop_10(out_conv_10)
        out_concat_11 = torch.cat((out_conv_10_crop, out_upconv_11), dim = 1)
        out_conv_11 = self.conv_11(out_concat_11)
        out_conv_11 = self.activation(out_conv_11)

        out_upconv_21 = self.upconv_21(out_conv_30)
        crop_20 = CenterCrop(size = out_upconv_21.size(dim=2))
        out_conv_20_crop = crop_20(out_conv_20)
        out_concat_21 = torch.cat((out_conv_20_crop, out_upconv_21), dim = 1)
        out_conv_21 = self.conv_21(out_concat_21)
        out_conv_21 = self.activation(out_conv_21)

        out_upconv_31 = self.upconv_31(out_conv_40)
        crop_30 = CenterCrop(size = out_upconv_31.size(dim=2))
        out_conv_30_crop = crop_30(out_conv_30)
        out_concat_31 = torch.cat((out_conv_30_crop, out_upconv_31), dim = 1)
        out_conv_31 = self.conv_31(out_concat_31)
        out_conv_31 = self.activation(out_conv_31)


        out_upconv_02 = self.upconv_02(out_conv_11)
        crop_00 = CenterCrop(size = out_upconv_02.size(dim=2))
        out_conv_00_crop = crop_00(out_conv_00)
        crop_01 = CenterCrop(size = out_upconv_02.size(dim=2))
        out_conv_01_crop = crop_01(out_conv_01)
        out_concat_02 = torch.cat((out_conv_00_crop, out_conv_01_crop, out_upconv_02), dim = 1)
        out_conv_02 = self.conv_02(out_concat_02)
        out_conv_02 = self.activation(out_conv_02)

        out_upconv_12 = self.upconv_12(out_conv_21)
        crop_10 = CenterCrop(size = out_upconv_12.size(dim=2))
        out_conv_10_crop = crop_10(out_conv_10)
        crop_11 = CenterCrop(size = out_upconv_12.size(dim=2))
        out_conv_11_crop = crop_11(out_conv_11)
        out_concat_12 = torch.cat((out_conv_10_crop, out_conv_11_crop, out_upconv_12), dim = 1)
        out_conv_12 = self.conv_12(out_concat_12)
        out_conv_12 = self.activation(out_conv_12)

        out_upconv_22 = self.upconv_22(out_conv_31)
        crop_20 = CenterCrop(size = out_upconv_22.size(dim=2))
        out_conv_20_crop = crop_20(out_conv_20)
        crop_21 = CenterCrop(size = out_upconv_22.size(dim=2))
        out_conv_21_crop = crop_21(out_conv_21)
        out_concat_22 = torch.cat((out_conv_20_crop, out_conv_21_crop, out_upconv_22), dim = 1)
        out_conv_22 = self.conv_22(out_concat_22)
        out_conv_22 = self.activation(out_conv_22)


        out_upconv_03 = self.upconv_03(out_conv_12)
        crop_00 = CenterCrop(size = out_upconv_03.size(dim=2))
        out_conv_00_crop = crop_00(out_conv_00)
        crop_01 = CenterCrop(size = out_upconv_03.size(dim=2))
        out_conv_01_crop = crop_01(out_conv_01)
        crop_02 = CenterCrop(size = out_upconv_03.size(dim=2))
        out_conv_02_crop = crop_02(out_conv_02)
        out_concat_03 = torch.cat((out_conv_00_crop, out_conv_01_crop, out_conv_02_crop, out_upconv_03), dim = 1)
        out_conv_03 = self.conv_03(out_concat_03)
        out_conv_03 = self.activation(out_conv_03)

        out_upconv_13 = self.upconv_13(out_conv_22)
        crop_10 = CenterCrop(size = out_upconv_13.size(dim=2))
        out_conv_10_crop = crop_10(out_conv_10)
        crop_11 = CenterCrop(size = out_upconv_13.size(dim=2))
        out_conv_11_crop = crop_11(out_conv_11)
        crop_12 = CenterCrop(size = out_upconv_13.size(dim=2))
        out_conv_12_crop = crop_12(out_conv_12)
        out_concat_13 = torch.cat((out_conv_10_crop, out_conv_11_crop, out_conv_12_crop, out_upconv_13), dim = 1)
        out_conv_13 = self.conv_13(out_concat_13)
        out_conv_13 = self.activation(out_conv_13)


        out_upconv_04 = self.upconv_04(out_conv_13)
        crop_00 = CenterCrop(size = out_upconv_04.size(dim=2))
        out_conv_00_crop = crop_00(out_conv_00)
        crop_01 = CenterCrop(size = out_upconv_04.size(dim=2))
        out_conv_01_crop = crop_01(out_conv_01)
        crop_02 = CenterCrop(size = out_upconv_04.size(dim=2))
        out_conv_02_crop = crop_02(out_conv_02)
        crop_03 = CenterCrop(size = out_upconv_04.size(dim=2))
        out_conv_03_crop = crop_03(out_conv_03)
        out_concat_04 = torch.cat((out_conv_00_crop, out_conv_01_crop, out_conv_02_crop, out_conv_03_crop, out_upconv_04), dim = 1)
        out_conv_04 = self.conv_04(out_concat_04)
        out_conv_04 = self.activation(out_conv_04)

        # Output<
        out = self.conv_final(out_conv_04)

        # Apply softmax along the channel dimension
        #

        # out = self.softmax(out)
        # out = torch.argmax(out, dim=1)

        # out = sftmx(out)

        # Apply argmax to find the class with the largest probability for each pixel.

        # One-hot encoding of the segmentation
        #out = one_hot(out.to(torch.int64), num_classes=9)
        #out = torch.permute(out, (0,3,1,2))


        return out




