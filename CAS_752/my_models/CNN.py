'''Convolutional Neural Network (CNN) model'''

import torch.nn as nn

class CNN_DQN(nn.Module):
    def __init__(self, game_size, num_actions):
        super(CNN_DQN, self).__init__()

    def forward(self, x):