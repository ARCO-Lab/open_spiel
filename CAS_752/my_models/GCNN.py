'''Group equivariant Convolutional Neural Network (G-CNN) model'''

from torch import nn

class GCNN_DQN(nn.Module):
    def __init__(self, game_size, num_actions):
        super(GCNN_DQN, self).__init__()

    def forward(self, x):
        