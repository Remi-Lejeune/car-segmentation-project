# Load functions
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, ConvTranspose2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import ReLU
import numpy as np
from torchvision.transforms import CenterCrop


image = rgb2grayscale(image) # Here we need some real code for converting the images to grayscale

height, width, channels = IMAGE_SHAPE # (256, 256, 1) 

# CNN parameters
conv_out_channels_1 = 16  # <-- Filters in first convolutional layer
conv_out_channels_2 = 32  # <-- Filters in second convolutional layer
kernel_size = 5      # <-- Kernel size
conv_stride = 1       # <-- Stride
conv_pad    = 1       # <-- Padding
 

# Keep track of features to output layer
features_out_cnn_1 = (16,int(62/2),int(62/2)) # <-- Size of the output from the first convolutional layer after maxpool layer
features_out_cnn_2 = (32, 14, 14) # <-- Size of the output from the second convolutional layer after maxpool layer
rnn_out_size = rnn_hidden_size    # Size of the output from the recurrrent layer

# Size of the total input to the linear layer after concatenating
features_cat_size =  np.prod(features_out_cnn_2) + rnn_out_size + 64+64   
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size = 2, stride = 2)

        # First set of convolutional layers
        self.conv_11 = Conv2d(in_channels = channels, 
                                out_channels = 64, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (254, 254, 64)

        self.conv_12 = Conv2d(in_channels = 64, 
                                out_channels = 64, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (252, 252, 64)

        
                        

        # Second set of convolutional layers
        self.conv_21 = Conv2d(in_channels = 64, 
                                out_channels = 128, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (124, 124, 128)


        self.conv_22 = Conv2d(in_channels = 128, 
                                out_channels = 128, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (122, 122, 128)

        # Third set of convolutional layers
        self.conv_31 = Conv2d(in_channels = 128, 
                                out_channels = 256, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (59, 59, 256)


        self.conv_32 = Conv2d(in_channels = 256, 
                                out_channels = 256, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (57, 57, 256)


        # Fourth set of convolutional layers
        self.conv_41 = Conv2d(in_channels = 256, 
                                out_channels = 512, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (26, 26, 512)


        self.conv_42 = Conv2d(in_channels = 512, 
                                out_channels = 512, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (24, 24, 512)
        

        self.upconv_4 = ConvTranspose2d(in_channels = 512, 
                                            out_channels = 256, 
                                            kernel_size = 2, 
                                            stride = 1, 
                                            padding = 0) # Output size from this layer (48, 48, 256)



        # Fith set of convolutional layers
        self.conv_51 = Conv2d(in_channels = 512, 
                                out_channels = 256, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (46, 46, 256)


        self.conv_52 = Conv2d(in_channels = 256, 
                                out_channels = 256, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (44, 44, 256)
        

        self.upconv_5 = ConvTranspose2d(in_channels = 256, 
                                            out_channels = 128, 
                                            kernel_size = 2, 
                                            stride = 1, 
                                            padding = 0) # Output size from this layer (88, 88, 128)


        # Sixth set of convolutional layers
        self.conv_61 = Conv2d(in_channels = 256, 
                                out_channels = 128, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (86, 86, 128)


        self.conv_62 = Conv2d(in_channels = 128, 
                                out_channels = 128, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (84, 84, 128)
        

        self.upconv_6 = ConvTranspose2d(in_channels = 128, 
                                            out_channels = 64, 
                                            kernel_size = 2, 
                                            stride = 1, 
                                            padding = 0) # Output size from this layer (168, 168, 64)


        # Seventh set of convolutional layers
        self.conv_71 = Conv2d(in_channels = 128, 
                                out_channels = 64, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (168, 168, 64)


        self.conv_72 = Conv2d(in_channels = 64, 
                                out_channels = 64, 
                                kernel_size = 3,
                                stride = 1,
                                padding = 0)  # Output size from this layer (166, 166, 64)

        self.conv_73 = Conv2d(in_channels = 64, 
                                out_channels = 2, 
                                kernel_size = 1,
                                stride = 1,
                                padding = 0)  # Output size from this layer (166, 166, 2)

        
    def forward(self, x_img, x_margin, x_shape, x_texture):
        features = []
        out = {}
        
        # First set of  convolutional layers 
        out_conv_1 = self.conv_11(x_img)        # Output size from this layer (254, 254, 64)
        out_conv_1 = self.ReLU(out_conv_1)      

        out_conv_1 = self.conv_12(out_conv_1)   # Output size from this layer (252, 252, 64)
        out_conv_1 = self.ReLU(out_conv_1)

        out_maxpool_1 = self.maxpool(out_conv_1) # Output size from this layer (126, 126, 64)

        # Second set of convolutional layers
        out_conv_2 = self.conv_21(out_maxpool_1) # Output size from this layer (124, 124, 128)
        out_conv_2 = self.ReLU(out_conv_2) 

        out_conv_2 = self.conv_22(out_conv_2)    # Output size from this layer (122, 122, 128)
        out_conv_2 = self.ReLU(out_conv_2)

        out_maxpool_2 = self.maxpool(out_conv_2) # Output size from this layer (61, 61, 128)

        # Third set of convolutional layers
        out_conv_3 = self.conv_31(out_maxpool_2) # Output size from this layer (59, 59, 256)
        out_conv_3 = self.ReLU(out_conv_3) 

        out_conv_3 = self.conv_32(out_conv_3)    # Output size from this layer (57, 57, 256)
        out_conv_3 = self.ReLU(out_conv_3)

        out_maxpool_3 = self.maxpool(out_conv_3) # Output size from this layer (28, 28, 256)

        # Fourth set of convolutional layers
        out_conv_4 = self.conv_41(out_maxpool_3) # Output size from this layer (26, 26, 512)
        out_conv_4 = self.ReLU(out_conv_4) 

        out_conv_4 = self.conv_42(out_conv_4)    # Output size from this layer (24, 24, 512)
        out_conv_4 = self.ReLU(out_conv_4)

        out_upconv_4 = self.upconv_4(out_conv_4) # Output size from this layer (48, 48, 256)

        # Crop the output from conv_3 to fit the output from upconv_4 and concatenate the two
        out_conv_3_cropped = CenterCrop(size = 48)(torch.permute(out_conv_3, (2,0,1)))
        out_conv_3_cropped = torch.permute(out_conv_3_cropped, (1,2,0))
        out_concat_4 = torch.cat((out_conv_3_cropped, out_upconv_4), dim = 2)  # Size (48, 48, 512)


        # Fith set of convolutional layers
        out_conv_5 = self.conv_51(out_concat_4)  # Output size from this layer (46, 46, 256)
        out_conv_5 = self.ReLU(out_conv_5) 

        out_conv_5 = self.conv_52(out_conv_5)    # Output size from this layer (44, 44, 256)
        out_conv_5 = self.ReLU(out_conv_5)

        out_upconv_5 = self.upconv_5(out_conv_5) # Output size from this layer (88, 88, 128)

        # Crop the output from conv_2 to fit the output from upconv_5 and concatenate the two
        out_conv_2_cropped = CenterCrop(size = 88)(torch.permute(out_conv_2, (2,0,1)))
        out_conv_2_cropped = torch.permute(out_conv_2_cropped, (1,2,0))
        out_concat_5 = torch.cat((out_conv_2_cropped, out_upconv_5), dim = 2)  # Size (88, 88, 256)


        # Sixth set of convolutional layers
        out_conv_6 = self.conv_61(out_concat_5)  # Output size from this layer (86, 86, 128)
        out_conv_6 = self.ReLU(out_conv_6) 

        out_conv_6 = self.conv_62(out_conv_6)    # Output size from this layer (84, 84, 128)
        out_conv_6 = self.ReLU(out_conv_6)

        out_upconv_6 = self.upconv_6(out_conv_6) # Output size from this layer (168, 168, 64)

        # Crop the output from conv_1 to fit the output from upconv_6 and concatenate the two
        out_conv_1_cropped = CenterCrop(size = 168)(torch.permute(out_conv_1, (2,0,1)))
        out_conv_1_cropped = torch.permute(out_conv_1_cropped, (1,2,0))
        out_concat_6 = torch.cat((out_conv_1_cropped, out_upconv_6), dim = 2)  # Size (168, 168, 128)


        # Seventh set of convolutional layers
        out_conv_7 = self.conv_71(out_concat_6)  # Output size from this layer (168, 168, 64)
        out_conv_7 = self.ReLU(out_conv_7) 

        out_conv_7 = self.conv_72(out_conv_7)    # Output size from this layer (166, 166, 64)
        out_conv_7 = self.ReLU(out_conv_7)

        out_conv_7 = self.conv_73(out_conv_7)    # Output size from this layer (166, 166, 2)

        out = out_conv_7

        return out

net = Net()
