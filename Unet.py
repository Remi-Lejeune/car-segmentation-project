# Load functions
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, ConvTranspose2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import ReLU, Sigmoid, Softmax
import numpy as np
from torchvision.transforms import CenterCrop

    
class Net(nn.Module):
    def __init__(self, im_height, im_width, im_channels, num_features, activationFun="Relu", maxpool_kernel_size=2, 
                maxpool_kernel_stride=2, conv_kernel_size = 3, conv_kernel_stride=1, conv_padding=0, conv_dilation=1, 
                conv_out_channels_0 = 64, upconv_kernel_size=2, upconv_kernel_stride=1, upconv_padding=0):
        super(Net, self).__init__()

        if activationFun=="Relu":
            self.activation = ReLU()
        elif activationFun == "Sigmoid":
            self.activation = Sigmoid()
        else:
            raise ValueError('Invalid activation function')

        self.maxpool = MaxPool2d(kernel_size = maxpool_kernel_size, stride = maxpool_kernel_stride)

        # First set of convolutional layers
        self.conv_11 = Conv2d(in_channels = im_channels, 
                                out_channels = conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (254, 254, 64)

        self.conv_12 = Conv2d(in_channels = conv_out_channels_0, 
                                out_channels = conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (252, 252, 64)

        
                        

        # Second set of convolutional layers
        self.conv_21 = Conv2d(in_channels = conv_out_channels_0, 
                                out_channels = 2*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (124, 124, 128)


        self.conv_22 = Conv2d(in_channels = 2*conv_out_channels_0, 
                                out_channels = 2*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (122, 122, 128)

        # Third set of convolutional layers
        self.conv_31 = Conv2d(in_channels = 2*conv_out_channels_0, 
                                out_channels = 4*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (59, 59, 256)


        self.conv_32 = Conv2d(in_channels = 4*conv_out_channels_0, 
                                out_channels = 4*conv_out_channels_0, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (57, 57, 256)


        # Fourth set of convolutional layers
        self.conv_41 = Conv2d(in_channels = 4*conv_out_channels_0, 
                                out_channels = 8*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (26, 26, 512)


        self.conv_42 = Conv2d(in_channels = 8*conv_out_channels_0, 
                                out_channels = 8*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (24, 24, 512)
        

        self.upconv_4 = ConvTranspose2d(in_channels = 8*conv_out_channels_0, 
                                            out_channels = 4*conv_out_channels_0, 
                                            kernel_size = upconv_kernel_size, 
                                            stride = upconv_kernel_stride, 
                                            padding = upconv_padding) # Output size from this layer (48, 48, 256)



        # Fith set of convolutional layers
        self.conv_51 = Conv2d(in_channels = 8*conv_out_channels_0, 
                                out_channels = 4*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (46, 46, 256)


        self.conv_52 = Conv2d(in_channels = 4*conv_out_channels_0, 
                                out_channels = 4*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (44, 44, 256)
        

        self.upconv_5 = ConvTranspose2d(in_channels = 4*conv_out_channels_0, 
                                            out_channels = 2*conv_out_channels_0, 
                                            kernel_size = upconv_kernel_size, 
                                            stride = upconv_kernel_stride, 
                                            padding = upconv_padding) # Output size from this layer (88, 88, 128)


        # Sixth set of convolutional layers
        self.conv_61 = Conv2d(in_channels = 4*conv_out_channels_0, 
                                out_channels = 2*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding) # Output size from this layer (86, 86, 128)


        self.conv_62 = Conv2d(in_channels = 2*conv_out_channels_0, 
                                out_channels = 2*conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding) # Output size from this layer (84, 84, 128)
        

        self.upconv_6 = ConvTranspose2d(in_channels = 2*conv_out_channels_0, 
                                            out_channels = conv_out_channels_0, 
                                            kernel_size = upconv_kernel_size, 
                                            stride = upconv_kernel_stride, 
                                            padding = upconv_padding) # Output size from this layer (168, 168, 64)


        # Seventh set of convolutional layers
        self.conv_71 = Conv2d(in_channels = 2*conv_out_channels_0, 
                                out_channels = conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (168, 168, 64)


        self.conv_72 = Conv2d(in_channels = conv_out_channels_0, 
                                out_channels = conv_out_channels_0, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (166, 166, 64)

        self.conv_73 = Conv2d(in_channels = conv_out_channels_0, 
                                out_channels = num_features, 
                                kernel_size = conv_kernel_size,
                                stride = conv_kernel_stride,
                                padding = conv_padding)  # Output size from this layer (166, 166, num_features)

        
    def forward(self, x_img):
        out = {}
        
        print("Data type of input: ", x_img.dtype)
        print("Shape of input: ", x_img.shape)

        # First set of  convolutional layers 
        out_conv_1 = self.conv_11(x_img)        # Output size from this layer (254, 254, 64)
        out_conv_1 = self.activation(out_conv_1)      

        out_conv_1 = self.conv_12(out_conv_1)   # Output size from this layer (252, 252, 64)
        out_conv_1 = self.activation(out_conv_1)

        out_maxpool_1 = self.maxpool(out_conv_1) # Output size from this layer (126, 126, 64)

        # Second set of convolutional layers
        out_conv_2 = self.conv_21(out_maxpool_1) # Output size from this layer (124, 124, 128)
        out_conv_2 = self.activation(out_conv_2) 

        out_conv_2 = self.conv_22(out_conv_2)    # Output size from this layer (122, 122, 128)
        out_conv_2 = self.activation(out_conv_2)

        out_maxpool_2 = self.maxpool(out_conv_2) # Output size from this layer (61, 61, 128)

        # Third set of convolutional layers
        out_conv_3 = self.conv_31(out_maxpool_2) # Output size from this layer (59, 59, 256)
        out_conv_3 = self.activation(out_conv_3) 

        out_conv_3 = self.conv_32(out_conv_3)    # Output size from this layer (57, 57, 256)
        out_conv_3 = self.activation(out_conv_3)

        out_maxpool_3 = self.maxpool(out_conv_3) # Output size from this layer (28, 28, 256)

        # Fourth set of convolutional layers
        out_conv_4 = self.conv_41(out_maxpool_3) # Output size from this layer (26, 26, 512)
        out_conv_4 = self.activation(out_conv_4) 

        out_conv_4 = self.conv_42(out_conv_4)    # Output size from this layer (24, 24, 512)
        out_conv_4 = self.activation(out_conv_4)

        out_upconv_4 = self.upconv_4(out_conv_4) # Output size from this layer (48, 48, 256)

        # Crop the output from conv_3 to fit the output from upconv_4 and concatenate the two
        out_conv_3_cropped = CenterCrop(size = out_upconv_4.size(dim=2))
        out_concat_4 = torch.cat((out_conv_3_cropped, out_upconv_4), dim = 1)  # Size (48, 48, 512)


        # Fith set of convolutional layers
        out_conv_5 = self.conv_51(out_concat_4)  # Output size from this layer (46, 46, 256)
        out_conv_5 = self.activation(out_conv_5) 

        out_conv_5 = self.conv_52(out_conv_5)    # Output size from this layer (44, 44, 256)
        out_conv_5 = self.activation(out_conv_5)

        out_upconv_5 = self.upconv_5(out_conv_5) # Output size from this layer (88, 88, 128)

        # Crop the output from conv_2 to fit the output from upconv_5 and concatenate the two
        out_conv_2_cropped = CenterCrop(size = out_upconv_5.size(dim=2))
        out_concat_5 = torch.cat((out_conv_2_cropped, out_upconv_5), dim = 1)  # Size (88, 88, 256)


        # Sixth set of convolutional layers
        out_conv_6 = self.conv_61(out_concat_5)  # Output size from this layer (86, 86, 128)
        out_conv_6 = self.activation(out_conv_6) 

        out_conv_6 = self.conv_62(out_conv_6)    # Output size from this layer (84, 84, 128)
        out_conv_6 = self.activation(out_conv_6)

        out_upconv_6 = self.upconv_6(out_conv_6) # Output size from this layer (168, 168, 64)

        # Crop the output from conv_1 to fit the output from upconv_6 and concatenate the two
        out_conv_1_cropped = CenterCrop(size = out_upconv_6.size(dim=2))
        out_concat_6 = torch.cat((out_conv_1_cropped, out_upconv_6), dim = 1)  # Size (168, 168, 128)


        # Seventh set of convolutional layers
        out_conv_7 = self.conv_71(out_concat_6)  # Output size from this layer (168, 168, 64)
        out_conv_7 = self.activation(out_conv_7) 

        out_conv_7 = self.conv_72(out_conv_7)    # Output size from this layer (166, 166, 64)
        out_conv_7 = self.activation(out_conv_7)

        out_conv_7 = self.conv_73(out_conv_7)   

        # Apply softmax along the channel dimension
        sftmx = Softmax(dim=1)
        out = sftmx(out)

        out = out_conv_7

        return out

#net = Net()
