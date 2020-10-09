import torch.nn as nn
import torch.nn.functional as F

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


def train_bi(bi_data_dic):
    # Perform train for each dataset in bi_data_dic
    ## Iterate over all possible H values
    ### Run k-fold cross-validation and average the total binary-cross entropy
    ### Store values of every H and CV score
    ### plot results
    ### Use max and output test score
    pass




