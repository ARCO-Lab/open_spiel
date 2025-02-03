'''Fully connected neural network model'''

import torch.nn as nn
import torch.nn.functional as F

class FC_DQN(nn.Module):
    def __init__(self, game_size, num_actions):
        '''
        game_size is size of input layer
        num_actions is size of output layer
        '''
        super(FC_DQN, self).__init__()
        self.layer1 = nn.Linear(game_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    