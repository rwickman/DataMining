import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.model_selection import KFold
from models import BinaryNetwork

class TrainerBN:
    def __init__(self, dataset, lr=1e-3, epochs=10, batch_size = 8):
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size

        # Combines Sigmoid with binary cross entropy loss
        self._loss_fn = nn.BCEWithLogitsLoss()
        
        self._H_list = np.arange(2, 10)#np.arange(2, 11)
        self._train_X = dataset["train_X"]
        self._train_Y = dataset["train_Y"][:, np.newaxis]
        self._test_X = dataset["test_X"]
        self._test_Y = dataset["test_Y"][:, np.newaxis]

        
    def k_fold_cv(self, n_splits=5):
        kf = KFold(n_splits=n_splits)

        # Iterate over all the possible number of units
        avg_val_losses = []
        for H in self._H_list:
            # Build the network
            bi_net = BinaryNetwork(self._train_X.shape[1], H)    
            optimizer = optim.Adam(bi_net.parameters(), lr=self._lr)
            

            # Run k-fold cross-validation
            fold_losses = []
            for train_idx, test_idx in kf.split(self._train_X):
                train_fold_X = self._train_X[train_idx]
                train_fold_Y = self._train_Y[train_idx]
                
            
                # Train over all the epochs
                for e_i in range(self._epochs):
                    print("Training on epoch ", e_i)
                    # Train on mini-batches
                    total_batch_loss = 0
                    trainloader = torch.utils.data.DataLoader(list(zip(train_fold_X, train_fold_Y)), batch_size = self._batch_size)
                    for train_batch in trainloader:
                        train_batch_X, train_batch_Y = train_batch

                        # Predict on batch
                        outputs = bi_net(train_batch_X.float())
                        # Compute loss on batch
                        batch_loss = self._loss_fn(outputs, train_batch_Y.float())
                        total_batch_loss += batch_loss.item()
                        # Compute gradient
                        batch_loss.backward()
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                    print("Batch Loss: ", total_batch_loss)
                
                
                # Compute fold loss
                outputs = bi_net(torch.from_numpy(self._train_X[test_idx]).float())
                fold_loss = self._loss_fn(outputs, torch.from_numpy(self._train_Y[test_idx]).float()).item()
                print("Test fold loss: ", fold_loss)
                fold_losses.append(fold_loss)
            print(fold_losses)
            avg_val_losses.append(sum(fold_losses)/len(fold_losses))
        
        print(avg_val_losses)
