import nibabel as nib
import os
from torch.utils.data import Dataset
import random
import numpy as np
import torch



class AD_MultiData(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, data_file, MRI_data_dict, PET_data_dict, PET_label_dict, SNP_data_dict, SNP_label_dict):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_file = data_file
        self.MRI_data_dict = MRI_data_dict
        self.PET_data_dict = PET_data_dict
        self.PET_label_dict = PET_label_dict
        self.SNP_data_dict = SNP_data_dict
        self.SNP_label_dict = SNP_label_dict


    def __len__(self):
        return sum(1 for line in open(self.data_file))

    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()

        data_ID = (lst[0])
        data_sMRI = self.MRI_data_dict[data_ID]
        data_PET = self.PET_data_dict[data_ID]
        label_PET =self.PET_label_dict[data_ID]
        data_SNP = self.SNP_data_dict[data_ID]
        label_SNP =self.SNP_label_dict[data_ID]


        data_label = lst[1]

        if data_label == '0':   #NC
            label = 0
        if data_label == '1':   #MCI 1
            label = 1
        elif data_label == '2':   #MCI 2
            label = 2
        elif data_label == '3':   #MCI 3
            label = 3
        elif data_label == '4':   #AD
            label = 4



        samples = []

        data_sMRI = CustomToTensor(data_sMRI)
        sample = {"sMRI": data_sMRI, "PET": data_PET, "PET label": label_PET,  "SNP": data_SNP, "SNP label": label_SNP, "label": label}
        samples.append(sample)

        random.shuffle(samples)
        return samples





def CustomToTensor(pic):

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((0,1,2,3)))
        img = img.float()

        return img