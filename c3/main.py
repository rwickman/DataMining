import numpy as np
import os
from trainer import TrainerBN

def load_bi_data():
    data_dir = "datasets/bi-class"
    data_files = os.listdir("datasets/bi-class")
    bi_data_dict = {}
    # Load every data file into bi_data_dict
    for data_file in data_files:
        bi_data_dict[data_file.split(".")[0]] = dict(
            np.load(os.path.join(data_dir, data_file)))
    return bi_data_dict

bi_data_dict = load_bi_data()

trainer_bn = TrainerBN(bi_data_dict["iris"], patience=15)
trainer_bn.k_fold_cv(5, 1)