import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.model_selection import KFold
from models import BinaryNetwork, MultiNetwork


class Trainer:
    def __init__(self,
                loss_fn,
                epochs=25,
                lr=1e-2,
                batch_size=16,
                early_stopping=True,
                patience=10,
                decay_rate=0.99,
                num_rand_init=1):
        self._loss_fn = loss_fn
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._early_stopping = early_stopping
        self._patience = patience
        self._decay_rate = decay_rate
        self._num_rand_init = num_rand_init
        

    def _train(self, model, optimizer, lr_scheduler, train_X, train_Y, val_X, val_Y):

        # Train over all the epochs
        if self._early_stopping:
            min_val_loss = None
            steps_past_min = 0
        print(self._epochs)
        for e_i in range(self._epochs):
            if e_i % 5 == 0:
                print("Training on epoch ", e_i)
            # Train on mini-batches
            total_batch_loss = 0
            trainloader = torch.utils.data.DataLoader(
                list(zip(train_X, train_Y)),
                batch_size = self._batch_size)
            for train_batch in trainloader:
                train_batch_X, train_batch_Y = train_batch
                # Predict on batch
                outputs = model(train_batch_X.float())
                # Compute loss on batch
                batch_loss = self._loss_fn(outputs, train_batch_Y.float())
                total_batch_loss += batch_loss.item()
                
                # Compute gradient
                batch_loss.backward()
                # Update weights
                optimizer.step()
                                
                # Test on validation set
                outputs = model(torch.from_numpy(val_X).float())
                # Compute loss on batch
                val_batch_loss = self._loss_fn(outputs, torch.from_numpy(val_Y).float()).item()
                optimizer.zero_grad()
                
                #print("lr_scheduler.get_lr(): ", lr_scheduler.get_lr())
                if self._early_stopping:
                    if min_val_loss == None or min_val_loss >= val_batch_loss:
                        min_val_loss = val_batch_loss
                        steps_past_min = 0
                    else:
                        steps_past_min += 1
                    
                    if steps_past_min >= self._patience:
                        print("EARLY STOPPING ON EPOCH ", e_i)
                        return

            lr_scheduler.step()


            print("Batch Loss: ", total_batch_loss)



class TrainerBN(Trainer):
    def __init__(self,
                dataset,
                H_list,
                epochs=25,
                lr=1e-2,
                batch_size=16,
                early_stopping=True,
                patience=10,
                decay_rate=0.99,
                num_rand_init=1):
        """Create TrainerBN to train and perform hyperparameter tuning.

        Args:
            dataset: the dictionary of numpy training and testing data
            H_list: list of H values giving the units for the hidden layer to test for on cross-validation
            lr: learning rate
            epochs: the maximum number of times to train over the dataset
            batch_size: the size of mini-batches
            early_stopping: should stop after validation loss does not decrease past min after patience steps
            patience: the number of steps to wait before stopping for early_stopping
            decay_rate: the decay rate to use for exponential learning rate decay
        
        """

        super().__init__(nn.BCEWithLogitsLoss(),
                        epochs,
                        lr,
                        batch_size,
                        early_stopping,
                        patience,
                        decay_rate,
                        num_rand_init)
        # Combines Sigmoid with binary cross entropy loss
        
        self._H_list = H_list
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
        kf = KFold(n_splits=n_splits, shuffle=True)
        # Iterate over all the possible number of units
        avg_val_losses = []
        for H in self._H_list:    
            # Run k-fold cross-validation for num_init times
            fold_losses = []
            for _ in range(num_init):
                min_fold_avg_loss = None
                for train_idx, test_idx in kf.split(self._train_X):
                    # Build the network
                    bi_net = BinaryNetwork(self._train_X.shape[1], H)    
                    optimizer = optim.Adam(bi_net.parameters(), lr=self._lr)
                    # The LR Decay Scheduler
                    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self._decay_rate)
                    # Train on current training partition
                    self._train(
                        bi_net,
                        optimizer,
                        lr_scheduler,
                        self._train_X[train_idx],
                        self._train_Y[train_idx],
                        self._train_X[test_idx],
                        self._train_Y[test_idx])
                    # Compute fold loss
                    outputs = bi_net(torch.from_numpy(self._train_X[test_idx]).float())
                    fold_loss = self._loss_fn(outputs, torch.from_numpy(self._train_Y[test_idx]).float()).item()
                    print("Test fold loss: ", fold_loss)
                    fold_losses.append(fold_loss)
                
                #print(fold_losses)
                # Update the minimum average loss received
                avg_val_loss = sum(fold_losses)/len(fold_losses)
                if not min_fold_avg_loss or avg_val_loss < min_fold_avg_loss:
                    min_fold_avg_loss = avg_val_loss
            
            avg_val_losses.append(min_fold_avg_loss)

        #print(avg_val_losses)
        return avg_val_losses

    def get_trained_BN(self, H, val_percent=0.1):
        """Get a trained neural network.

        Args:
            H: the number of units in the hidden layers
            val_percent: the percentage of training data to use for the validation set
        """
        # Build the network
        bi_net = BinaryNetwork(self._train_X.shape[1], H)    
        optimizer = optim.Adam(bi_net.parameters(), lr=self._lr)
        # The LR Decay Scheduler
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self._decay_rate)
        
        # Split into random training and validation indices
        val_idxs = np.random.choice(
            self._train_X.shape[0],
            size=int(self._train_X.shape[0] * val_percent),
            replace=False)
        train_idxs = np.array([idx for idx in np.arange(self._train_X.shape[0]) if idx not in  val_idxs])

        #self._train_on_fold(bi_net, optimizer, train_idxs, val_idxs, lr_scheduler)
        self._train(
            bi_net,
            optimizer,
            lr_scheduler,
            self._train_X[train_idxs],
            self._train_Y[train_idxs],
            self._train_X[val_idxs],
            self._train_Y[val_idxs])
        # Get all the different losses
        # Get the final training loss
        outputs = bi_net(torch.from_numpy(self._train_X[train_idxs]).float())
        train_loss = self._loss_fn(outputs, torch.from_numpy(self._train_Y[train_idxs]).float()).item()
        
        # Get the final validation loss
        outputs = bi_net(torch.from_numpy(self._train_X[val_idxs]).float())
        val_loss = self._loss_fn(outputs, torch.from_numpy(self._train_Y[val_idxs]).float()).item()

        # Get the test loss
        outputs = bi_net(torch.from_numpy(self._test_X).float())
        test_loss = self._loss_fn(outputs, torch.from_numpy(self._test_Y).float()).item()
        
        losses = {
            "train" : train_loss,
            "validation" : val_loss,
            "test" : test_loss
        }
        
        print("Test BCE Loss: ", test_loss)
        return bi_net, losses


class TrainerMulti(Trainer):
    def __init__(self,
                image_data,
                L1_list,
                L2_list,
                epochs=25,
                lr=1e-2,
                batch_size=16,
                early_stopping=True,
                patience=10,
                decay_rate=0.99,
                num_rand_init=1):
        super().__init__(nn.CrossEntropyLoss(),
                        epochs,
                        lr,
                        batch_size,
                        early_stopping,
                        patience,
                        decay_rate,
                        num_rand_init)
        self._image_data = image_data
        self._L1_list = L1_list
        self._L2_list = L2_list
        self._batch_size = batch_size
        self._epochs = epochs

        self._loss_fn = nn.CrossEntropyLoss()
    
    def cross_validation(self, num_init=1, val_percent=0.2):
        """Perform cross-validation to hyperparamater tune L1 and L2.
        """

        val_losses = []
        # Split the data into training and validation
        train_X, train_Y, val_X, val_Y = self._train_val_split()

        for L1 in self._L1_list:
            for L2 in self._L2_list:
                best_val_loss = None
                for _ in range(num_init):
                    multi_net = MultiNetwork(train_X.shape[1], L1, L2)
                    pass




    def _train_val_split(self, val_percent=0.2):
        val_idxs = np.random.choice(
            self._image_data.train_X.shape[0],
            size=int(self._image_data.train_X.shape[0] * val_percent),
            replace=False)
        train_idxs = np.array(
            [idx for idx in np.arange(self._image_data.train_X.shape[0]) if idx not in val_idxs])
        return self._image_data.train_X[train_idxs], \
            self._image_data.train_Y[0, train_idxs], \
            self._image_data.train_X[val_idxs], \
            self._image_data.train_Y[0, val_idxs]

        