import nibabel as nib
import numpy as np
import os


def load_nii_from_txt(data_file, root_dir_sMRI, root_dir_PET):

    data_IDs = read_sample_names_from_txt(data_file)
    sMRI_data_dict = {}
    PET_data_dict = {}
    PET_label_dict = {}

    for data_ID in data_IDs:

        data_ID = data_ID.split("\t")[0]
        data_ID_split = data_ID.split("_")
        sample_ID = "_".join(data_ID_split[:3])  # '002_S_0295'
        timepoint = data_ID_split[-1]  # 'm60', 'bl', 'sc'

        # sMRI
        img_name_sMRI = f'mwp1ADNI_{sample_ID}_MR_{timepoint}.nii'
        image_path_sMRI = os.path.join(root_dir_sMRI, img_name_sMRI)
        np_sMRI = load_MRI_to_np(image_path_sMRI)
        sMRI_data_dict[data_ID] = np_sMRI

        # PET
        if timepoint in ['sc', 'bl']:
            img_name_PET = f'rADNI_{sample_ID}_PT_AV45_bl.nii'
        else:
            img_name_PET = f'rADNI_{sample_ID}_PT_AV45_{timepoint}.nii'

        image_path_PET = os.path.join(root_dir_PET, img_name_PET)
        if os.path.isfile(image_path_PET):
            np_PET = load_PET_to_np(image_path_PET)
            label_PET = 1
        else:
            np_PET = np.zeros((1, 91, 109, 91))
            label_PET = 0

        PET_data_dict[data_ID] = np_PET
        PET_label_dict[data_ID] = label_PET

    return sMRI_data_dict, PET_data_dict, PET_label_dict




def read_sample_names_from_txt(txt_file_path):

    with open(txt_file_path, 'r') as file:
        sample_names = file.read().splitlines()

    return sample_names


def load_MRI_to_np(file_path):

    nii_data = nib.load(file_path)
    nii_data = MRI_preprocess(nii_data.get_fdata())
    np_data = np.array(nii_data)

    return np_data


def load_PET_to_np(file_path):

    nii_data = nib.load(file_path)
    nii_data = PET_preprocess(nii_data.get_fdata())
    np_data = np.array(nii_data)

    return np_data


def MRI_preprocess(image):

  MRI = np.zeros((1, 100, 120, 100))
  x, y, z = 9, 10, 5
  MRI[0] = image[x:x + 100, y:y + 120, z:z + 100]

  return MRI


def PET_preprocess(image):

  PET = np.expand_dims(image, axis=0)
  PET = np.nan_to_num(PET, nan=0)

  return PET