# C-3. Implement MLP with PyTorch
## What Is This?
This uses a one layer MLP to perform binary classification fit on several different datasets. It also uses a two layer MLP to perform multi-class classification on images. Both of the models were built and trained using PyTorch and perform hyperparameter tuning to find the best number of hidden units. It will plot the results using Matplotlib.

## How to Run

1. Clone this entire repository.
2. Run the following command to run the Python3 code:

    ```shell
    python3 main.py
    ```

    If on Windows, you may need to run this instead:
    
    ```shell
    python main.py
    ```

## Change Settings
You can configure the settings by updating main.py. Look at trainer.py and the TrainerBN and TrainerMulti classes for all the different parameters.

## Report
The report of the experiment setup and results can be found under docs/Report.pdf.