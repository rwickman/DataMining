import numpy as np
import math, os 
from trainer import TrainerBN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_bi_data():
    data_dir = "datasets/bi-class"
    data_files = os.listdir("datasets/bi-class")
    bi_data_dict = {}
    # Load every data file into bi_data_dict
    for data_file in data_files:
        bi_data_dict[data_file.split(".")[0]] = dict(
            np.load(os.path.join(data_dir, data_file)))
    return bi_data_dict

def scale_dataset(dataset):
    scaler = StandardScaler().fit(dataset["train_X"])
    dataset["train_X"] = scaler.transform(dataset["train_X"])
    dataset["test_X"] = scaler.transform(dataset["train_X"])

def _plot_H_loss(val_losses_dict, H_list):
    fig, axs = plt.subplots(math.ceil(len(val_losses_dict)/2), 2)
    for i, name in enumerate(val_losses_dict):
        axs[i//2][i%2].set_title("BCE Validation Loss vs H for {} Dataset".format(name))
        axs[i//2][i%2].set_xlabel("H")
        axs[i//2][i%2].set_ylabel("BCE Validation Loss")
        axs[i//2][i%2].set_xticks(H_list)
        axs[i//2][i%2].grid(True)
        axs[i//2][i%2].plot(H_list, val_losses_dict[name])
    fig.suptitle("Binary Classification Neural Network H Hyperparameter Tuning", fontsize=24)
    fig.subplots_adjust(hspace=0.40)
    if math.ceil(len(val_losses_dict)/2) * 2 > len(val_losses_dict):
        fig.delaxes(axs[math.ceil(len(val_losses_dict)/2)-1][1])
    plt.show()

def main():
    # Perform hyperparameter tuning on number for
    # the number units in the hidden layer for
    # binary classification tasks
    H_list = np.arange(1,11)
    val_losses_dict = {}
    bi_data_dict = load_bi_data()
    for name, dataset in bi_data_dict.items():
        scale_dataset(dataset)
        trainer_bn = TrainerBN(dataset, H_list, patience=30, epochs=1)
        val_losses_dict[name] = trainer_bn.k_fold_cv(5, 1)

    _plot_H_loss(val_losses_dict, H_list)

if __name__ == "__main__":
    main()