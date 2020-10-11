import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BinaryNetwork(nn.Module):
    def __init__(self, input_size, H):
        """Create the one layer neural network used for binary classification.

        Args:
            input_size: the number of features.
            H: the number of hidden units.
        """
        super().__init__()
        self.hidden = nn.Linear(input_size, H)
        self.out = nn.Linear(H, 1)
    

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

class MultiNetwork(nn.Module):
    def __init__(self, input_size, L1, L2):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, L1)
        self.hidden_1 = nn.Linear(L1, L2)
        self.out = nn.Linear(L2, 10)
    
    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.self.out(x)
        return x

