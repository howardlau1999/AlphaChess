import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

# 3x3 Convolution
def conv3x3(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)

# conv + batch_nor + [ relu ]
def convBlock(in_channels, out_channels, non_linearity, kernel_size=3, padding=1):
    elems = [ conv3x3(in_channels, out_channels, kernel_size=kernel_size, padding=padding), 
            nn.BatchNorm2d(out_channels) ]
    if non_linearity is not None:
        elems.append( non_linearity )
    return nn.Sequential(*elems)

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = convBlock(channels, channels, self.relu)
        self.conv2 = convBlock(channels, channels, None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x

class View(nn.Module):

    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(*self.size)

class AlphaChessModel(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, residual_blocks, board_size=(9, 10)):
        super(AlphaChessModel, self).__init__()

        # First convolutional block
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = convBlock(in_channels, hidden_channels, self.relu)

        # Residual Blocks Tower
        res_blocks = [ResidualBlock(hidden_channels) for i in range(residual_blocks)]
        self.residual_blocks = nn.Sequential(*res_blocks)

        # policy head
        # the policy should be a 10 x 9 x 96 tensor during training
        # and the output is the logit probabilities
        self.policy_head = nn.Sequential( convBlock(hidden_channels, hidden_channels, self.relu),
                conv3x3(hidden_channels, out_channels, bias=True) )
        # value head
        self.policy_size = (out_channels, board_size[0], board_size[1])
        self.value_head = nn.Sequential( convBlock(hidden_channels, 1, self.relu, kernel_size=1, padding=0),
                View(-1, board_size[0] * board_size[1]),
                nn.Linear(board_size[0] * board_size[1], 256),
                self.relu,
                nn.Linear(256, 1),
                nn.Tanh())

    def forward(self, x):
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    def loss(self, p, v, target_p, target_v):
        import torch.nn.functional as F
        return F.cross_entropy(p.view(p.size(0), -1), target_p) + F.mse_loss(v, target_v) 