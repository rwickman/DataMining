import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.model_selection import KFold
from models import BinaryNetwork

class TrainerBN:
    def __init__(self, 
                dataset,
                lr=1e-3,
                epochs=25,
                batch_size=8,
                early_stopping=True,
                patience=10,
                num_rand_init=1):
        """Create TrainerBN to train and perform hyperparameter tuning.

        Args:
            dataset: the dictionary of numpy training and testing data
            lr: learning rate
            epochs: the maximum number of times to train over the dataset
            batch_size: the size of mini-batches
            early_stopping: should stop after validation loss does not decrease past min after patience steps
            patience: the number of steps to wait before stopping for early_stopping
        """
        
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        # 
        self._early_stopping = early_stopping
        self._patience = patience

        
        self._num_rand_init = num_rand_init

        # Combines Sigmoid with binary cross entropy loss
        self._loss_fn = nn.BCEWithLogitsLoss()
        
        self._H_list = np.arange(2, 10)#np.arange(2, 11)
        self._train_X = dataset["train_X"]
        self._train_Y = dataset["train_Y"][:, np.newaxis]
        self._test_X = dataset["test_X"]
        self._test_Y = dataset["test_Y"][:, np.newaxis]

        
    def k_fold_cv(self, n_splits=5, num_init=1):
        """Perfrom k-fold cross-validation.
        
        Args:
            n_splits: the amount of partition of the training dataset
            num_init: the amount of times to test one hyperparameter.
        """
        kf = KFold(n_splits=n_splits)

        # Iterate over all the possible number of units
        avg_val_losses = []
        for H in self._H_list:
            
            # Run k-fold cross-validation for num_init times
            fold_losses = []
            for _ in range(num_init):
                min_avg_loss = None
                for train_idx, test_idx in kf.split(self._train_X):
                    # Build the network
                    bi_net = BinaryNetwork(self._train_X.shape[1], H)    
                    optimizer = optim.Adam(bi_net.parameters(), lr=self._lr)
                    
                    # Train on current training partition
                    self._train_fold(bi_net, optimizer, train_idx, test_idx)
                    
                    # Compute fold loss
                    outputs = bi_net(torch.from_numpy(self._train_X[test_idx]).float())
                    fold_loss = self._loss_fn(outputs, torch.from_numpy(self._train_Y[test_idx]).float()).item()
                    print("Test fold loss: ", fold_loss)
                    fold_losses.append(fold_loss)
                
                print(fold_losses)
                # Update the minimum average loss received
                avg_val_loss = sum(fold_losses)/len(fold_losses)
                if not min_avg_loss or avg_val_loss < min_avg_loss:
                    min_avg_loss = avg_val_loss
            
            avg_val_losses.append(min_avg_loss)

        print(avg_val_losses)

    def _train_fold(self, bi_net, optimizer, train_idx, test_idx):
        train_fold_X = self._train_X[train_idx]
        train_fold_Y = self._train_Y[train_idx]
        test_fold_X = self._train_X[test_idx]
        test_fold_Y = self._train_Y[test_idx]

        # Train over all the epochs
        if self._early_stopping:
            min_val_loss = None
            steps_past_min = 0
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
                                
                # Test on validation set
                outputs = bi_net(torch.from_numpy(test_fold_X).float())
                # Compute loss on batch
                val_batch_loss = self._loss_fn(outputs, torch.from_numpy(test_fold_Y).float()) 
                
                optimizer.zero_grad()
                if self._early_stopping:
                    if min_val_loss == None or min_val_loss >= val_batch_loss:
                        min_val_loss = val_batch_loss
                        steps_past_min = 0
                    else:
                        steps_past_min += 1
                    
                    if steps_past_min >= self._patience:
                        print("EARLY STOPPING ON EPOCH ", e_i)
                        return


            print("Batch Loss: ", total_batch_loss)