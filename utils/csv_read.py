import torch
import numpy as np

def load_csv_from_txt(data_file, dataset_SNP):

    data_IDs = read_data_IDs_from_txt(data_file)
    SNP_data_dict = {}
    SNP_label_dict ={}

    for data_ID in data_IDs:

        data_ID = data_ID.split("\t")[0]
        data_ID_split = data_ID.split("_")
        sample_ID = "_".join(data_ID_split[:3])  # '002_S_0295'


        if sample_ID in dataset_SNP.index:
            data_SNP = dataset_SNP.loc[sample_ID].values
            label_SNP = 1

        else:
            data_SNP = np.zeros(1549)
            label_SNP = 0


        SNP_data_np = text_preprocess(data_SNP)
        SNP_data_dict[data_ID] = SNP_data_np
        SNP_label_dict[data_ID] = label_SNP

    return SNP_data_dict, SNP_label_dict



def text_preprocess(text):

    for i in range(len(text)):
        text[i] = float(text[i])

    if isinstance(text, np.ndarray):
        text = list(text)
        text = torch.Tensor(text)

    return text


def read_data_IDs_from_txt(txt_file_path):

    with open(txt_file_path, 'r') as file:
        data_IDs = file.read().splitlines()

    return data_IDs