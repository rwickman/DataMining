import numpy as np
import math, os 
from trainer import TrainerBN, TrainerMulti
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from dataclasses import dataclass

@dataclass
class ImageData:
    train_X: np.ndarray = None
    train_Y: np.ndarray = None
    test_X: np.ndarray = None
    test_Y: np.ndarray = None


def load_bi_data():
    data_dir = "datasets/bi-class"
    data_files = os.listdir(data_dir)
    bi_data_dict = {}
    # Load every data file into bi_data_dict
    for data_file in data_files:
        bi_data_dict[data_file.split(".")[0]] = dict(
            np.load(os.path.join(data_dir, data_file)))
    return bi_data_dict


def load_multi_data():
    data_dir = "datasets/multi-class"
    data_files = os.listdir(data_dir)
    image_data = ImageData()

    for data_file in data_files:
        mat_dict = loadmat(os.path.join(data_dir, data_file))
        if "train_images" in mat_dict:
            image_data.train_X = mat_dict["train_images"]
        elif "test_images" in mat_dict:
            image_data.test_X = mat_dict["test_images"]
        elif "train_labels" in mat_dict:
            image_data.train_Y = mat_dict["train_labels"]
        elif "test_labels" in mat_dict:
            image_data.test_Y = mat_dict["test_labels"]

    return image_data

def scale_dataset(dataset):
    if isinstance(dataset, ImageData):
        scaler = StandardScaler().fit(dataset.train_X)
        dataset.train_X = scaler.transform(dataset.train_X)
        dataset.test_X = scaler.transform(dataset.test_X)
    else:
        scaler = StandardScaler().fit(dataset["train_X"])
        dataset["train_X"] = scaler.transform(dataset["train_X"])
        dataset["test_X"] = scaler.transform(dataset["test_X"])

def plot_H_loss(val_losses_dict, H_list):
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

def plot_best_losses(best_net_losses_dict):
    fig, axs = plt.subplots(math.ceil(len(best_net_losses_dict)/2), 2)
    fig.suptitle("BCE Loss", fontsize=24)
    for i, name in enumerate(best_net_losses_dict):
        axs[i//2][i%2].set_title("BCE Losses for {} Dataset".format(name))
        axs[i//2][i%2].set_ylabel("BCE Loss")

        loss_type = []
        losses = []
        for k,v in best_net_losses_dict[name].items():
            loss_type.append(k)
            losses.append(v)
        x_pos = np.arange(len(losses))
        axs[i//2][i%2].bar(loss_type, losses, color=["blue", "orange", "red"], zorder=3)
        axs[i//2][i%2].grid(True, zorder=0)
        
        #axs[i//2][i%2].set_xticklabels(loss_type)
    
    fig.subplots_adjust(hspace=0.40)
    if math.ceil(len(best_net_losses_dict)/2) * 2 > len(best_net_losses_dict):
        fig.delaxes(axs[math.ceil(len(best_net_losses_dict)/2)-1][1])

def plot_bn_accs(best_results):
    fig, axs = plt.subplots(math.ceil(len(best_results)/2), 2)
    fig.suptitle("Binary Accuracy", fontsize=24)
    for i, name in enumerate(best_results):
        axs[i//2][i%2].set_title("Accuracy for {} Dataset".format(name))
        axs[i//2][i%2].set_ylabel("Accuracy")
        set_type = ["tran", "validation", "test"]
        accs = [
            best_results[name].train_accuracy,
            best_results[name].val_accuracy,
            best_results[name].test_accuracy]
        axs[i//2][i%2].bar(set_type, accs, color=["blue", "orange", "red"],  zorder=3)
        axs[i//2][i%2].grid(True, zorder=0)

    fig.subplots_adjust(hspace=0.40)
    if math.ceil(len(best_results)/2) * 2 > len(best_results):
        fig.delaxes(axs[math.ceil(len(best_results)/2)-1][1])

def plot_roc(roc_dict):
    pass

def plot_L1_L2_loss(param_val_dict):
    param_tuples = []
    losses = []
    accs = []
    
    # Get all of the the param combs, values, and accuracies
    for k,v in param_val_dict.items():
        param_tuples.append(str(k))
        losses.append(v[0])
        accs.append(v[1])
    
    fig, axs = plt.subplots(2)
    fig.suptitle("Multi-class Classification Neural Network (L1, L2) Hyperparameter Tuning", fontsize=24)
    axs[0].set_title("CE Validation Loss vs (L1,L2)", fontsize=16)
    axs[0].set_xlabel("(L1,L2)", fontsize=16)
    axs[0].set_ylabel("CE Validation Loss", fontsize=16)
    axs[0].grid(True)
    axs[0].plot(param_tuples, losses)

    axs[1].set_title("Accuracy vs (L1,L2)", fontsize=16)
    axs[1].set_xlabel("(L1,L2)", fontsize=16)
    axs[1].set_ylabel("Accuracy", fontsize=16)
    axs[1].grid(True)
    axs[1].plot(param_tuples, accs)


def plot_multi_acc(best_results):
    accs = [
            best_results.train_accuracy,
            best_results.val_accuracy,
            best_results.test_accuracy]
    set_type = ["tran", "validation", "test"]
    
    fig, axs = plt.subplots(1)
    fig.suptitle("Multi-class Accuracy", fontsize=24)
    axs.set_ylabel("Accuracy", fontsize=16)
    axs.bar(set_type, accs, color=["blue", "orange", "red"],  zorder=3)
    axs.grid(True, zorder=0)
    

def run_bn():
    # Perform hyperparameter tuning on number for the
    # number units in the hidden layer for binary classification tasks
    H_list = np.arange(1,11)
    val_losses_dict = {}
    best_net_losses_dict = {}
    best_results = {}
    bi_data_dict = load_bi_data()
    for name, dataset in bi_data_dict.items():
        scale_dataset(dataset)
        trainer_bn = TrainerBN(dataset, H_list, batch_size=16, patience=10, epochs=100)
        val_losses_dict[name] = trainer_bn.k_fold_cv(5, 3)

        # Get the best H value and train the network on it
        best_H = H_list[np.argmin(val_losses_dict[name])]
        print("BEST H: ", best_H)
        bi_net, losses, results = trainer_bn.get_trained_BN(best_H)
        best_net_losses_dict[name] = losses
        best_results[name] = results

    plot_best_losses(best_net_losses_dict)
    plot_H_loss(val_losses_dict, H_list)
    plot_bn_accs(best_results)
    plt.show()
    return best_results

def run_mn():
    image_data = load_multi_data()
    scale_dataset(image_data)
    L1_list = [50, 75, 100]
    L2_list = [10, 15, 20]
    #trainer = TrainerMulti(image_data, L1_list, L2_list, batch_size=256, patience=10, epochs=100)
    trainer = TrainerMulti(image_data, L1_list, L2_list, batch_size=256, patience=10, epochs=1)
    param_val_dict, results = trainer.cross_validation(num_init=1)
    #print(results)
    #print("MIN VALIDATION LOSS: ", min_val_loss, " BEST PARAMETERS: ", best_params)
    plot_L1_L2_loss(param_val_dict)
    plot_multi_acc(results)
    plt.show()
    return results


def main():
    #bn_best_results = run_bn()
    #print("\nBinary Classification Results: ", bn_best_results)
    mn_results = run_mn()
    print("\nMulti-class Classification Results: ", mn_results)

if __name__ == "__main__":
    main()