import glob

import numpy as np
from IPython.core.display import clear_output
from skimage import io
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, Parameter, init
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

import data_utils

data = []
i = 0

#
# # normalize the inputs
# x_train.div_(255)
# x_valid.div_(255)
# x_test.div_(255)


# Read the images
for np_name in glob.glob('carseg_data/clean_data/*.np[yz]'):
    data.append(np.load(f'{np_name}'))

    i += 1
    if i == 100:
        break
data = np.array(data)
print(data.shape)

np.random.shuffle(data)

x_train = data[:int(0.8*data.shape[0])]
x_valid = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]
x_test = data[int(0.9*data.shape[0]):]



# for i in range(100):
#     data[i] = np.load(f'carseg_data/clean_data/{i}.npy')


# plt.imshow(data[3])
# plt.show()

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

batch_size = 32
IMAGE_SHAPE = (3, 256, 256)


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


#Hyperparameters
num_classes = 10
num_l1 = 512
num_features = 262144

# define network
class Net(nn.Module):

    def __init__(self, num_features, num_hidden, num_output):
        super(Net, self).__init__()
        # input layer
        self.linear = nn.Linear(256 * 256, 256 * 256)
        # define activation function in constructor
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x.reshape(32, 256*256), dtype=torch.float32)
        print("tensor")
        x = self.linear(x)
        print("linear done")
        x = self.activation(x)
        print("activation done")

        return x


net = Net(num_features, num_l1, num_classes)

LEARNING_RATE = 0.002
criterion = nn.CrossEntropyLoss()  # <-- Your code here.

# weight_decay is equal to L2 regularization
optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def accuracy(ys, ts):
    predictions = torch.max(ys, 1)[1]
    correct_prediction = torch.eq(predictions, ts)
    return torch.mean(correct_prediction.float())


_img_shape = tuple([batch_size] + list(IMAGE_SHAPE))
_feature_shape = (batch_size)


def randnorm(size):
    return np.random.normal(0, 1, size).astype('float32')


# # dummy data
# _x_image = get_variable(Variable(torch.from_numpy(randnorm(_img_shape))))
#
# # test the forward pass
# output = net(x_img=_x_image, x_margin=_x_margin, x_shape=_x_shape, x_texture=_x_texture)
# output['out']
#
# # Setup settings for training
# VALIDATION_SIZE = 0.1  # 0.1 is ~ 100 samples for validation
# max_iter = 3000
# log_every = 100
# eval_every = 100


# Function to get label
def get_labels(batch):
    return get_variable(Variable(torch.from_numpy(batch['ts']).long()))


# Function to get input
def get_input(batch):
    return {
        'x_img': get_variable(Variable(torch.from_numpy(batch['images'])))
    }




# setting hyperparameters and gettings epoch sizes
batch_size = 32
num_epochs = 200
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

# setting up lists for handling loss/accuracy
train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
losses = []

def in_range(i, size):
    if i * size > num_samples_train:
        return False
    if (i + 1) * size > num_samples_train:
        return False

    return True


get_slice = lambda i, size: range(i * size, (i + 1) * size)

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):

        if not in_range(i, batch_size):
            break

        torch.nn.Dropout()
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(torch.tensor(x_train[slce][:,0,:,:]))
        print("output done")

        # compute gradients given loss
        target_batch = torch.tensor(x_train[slce][:,3,:,:])
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)

    net.eval()
    ### Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):

        if not in_range(i, batch_size):
            break

        slce = get_slice(i, batch_size)
        output = net(torch.tensor(x_train[slce][:,0,:,:]))

        preds = torch.max(output, 1)[1]

        train_targs += list(x_train[slce][:,3,:,:])
        train_preds += list(preds.data.numpy())

    ### Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):

        if not in_range(i, batch_size):
            break

        slce = get_slice(i, batch_size)

        output = net(torch.tensor(x_train[slce][:,0,:,:]))
        preds = torch.max(output, 1)[1]
        val_targs += list(x_train[slce][:,3,:,:].numpy())
        val_preds += list(preds.data.numpy())

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)

    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
            epoch + 1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accucary', 'Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')




