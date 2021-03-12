# Code taken from:
# https://github.com/neverparadise/MineRL_CapstonDesign

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# Object representing the device on which a torch.Tensor is or will be allocated
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.nn.module -> Base class for all neural network modules
class DQN(nn.Module):
    def __init__(self):
        self.num_actions = 3
        super(DQN, self).__init__()

        # Applies a 2D convolution over an input signal composed of several input planes.
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # nn.con2d(
        #           in_channels,         - Number of channels in the input image
        #           out_channels,        - Number of channels produced by the convolution, https://en.wikipedia.org/wiki/Convolution
        #           kernel_size,         - Size of the convolving kernel,                  https://en.wikipedia.org/wiki/Kernel_(image_processing)
        #
        #           [Remaining parameters are optinal]
        #           stride=1,            - controls the stride for the cross-correlation, a single number or a tuple.
        #           padding=0,           - controls the amount of implicit padding on both sides for padding number of points for each dimension.
        #           dilation=1,          - controls the spacing between the kernel points; also known as the à trous algorithm. https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        #           groups=1,            - controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups.
        #           bias=True,           - If True, adds a learnable bias to the output. Default: True
        #           padding_mode='zeros' - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        #           ) 
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)

        # orch.nn.BatchNorm2d() -> Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        # https://arxiv.org/abs/1502.03167
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(64, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)

        linear_input_size = convw * convw * 64
        self.head = nn.Linear(linear_input_size, self.num_actions)

    def forward(self, x):
        if(len(x.shape) < 4):
            x = x.unsqueeze(0).to(device=device)

        # torch.nn.functional.relu() -> Applies the rectified linear unit function element-wise
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.head(x.view(x.size(0), -1)))  # view"is" numpy"of" reshape "equal to".
        #x = F.softmax(x, dim=0)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)

        # Adds randomness
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,self.num_actions-1)
        else:
            #print(out)
            return torch.argmax(out)

