#Used to perform t-SNE dimensionality reduction on the multimodal fused features $Z_f$ in the latent space
#The MCI staging results need to be combined with the AD and NC samples and saved into `result_set.txt`.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import VAE_all
from utils.nii_read import load_nii_from_txt
from utils.csv_read import load_csv_from_txt
from utils.Data_label_preprocess_stage import AD_MultiData

device = torch.device("cpu")

model = VAE_all()
model.to(device)

z_f_all = []
label_all = []

IMG_PATH_sMRI = 'data/mwp1_nii/'
IMG_PATH_PET = 'data/PET/'
dataset_SNP = pd.read_csv('data/Data_SNP.csv', index_col='ID')
dataset_PATH ="./result_set_all.txt"

nii_data_dict, PET_data_dict, PET_label_dict = load_nii_from_txt(dataset_PATH, IMG_PATH_sMRI, IMG_PATH_PET)
SNP_data_dict, SNP_label_dict = load_csv_from_txt(dataset_PATH, dataset_SNP)

TESTING_PATH = "./result_set_all.txt"
dset_test = AD_MultiData(TESTING_PATH, nii_data_dict, PET_data_dict, PET_label_dict, SNP_data_dict, SNP_label_dict)
test_loader = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)

model.eval()
model_weight_path = "model/MCI_STH.pth"
model.load_state_dict(torch.load(model_weight_path))

with torch.no_grad():
    for test_data in test_loader:
        for data_dic in test_data:
            sMRI, PET, label_PET, SNP, label_SNP, label = (
                Variable(data_dic['sMRI']).to(device),
                Variable(data_dic['PET']).to(device),
                Variable(data_dic['PET label']).to(device),
                Variable(data_dic['SNP']).to(device),
                Variable(data_dic['SNP label']).to(device),
                Variable(data_dic['label']).to(device)
            )


            mask_PET = torch.zeros(label.shape).to(device)
            new_label_PET = mask_PET
            mask_SNP = torch.zeros(label.shape).to(device)
            new_label_SNP = mask_SNP


            sMRI_recon_x, sMRI_mu, sMRI_logvar, sMRI_z,\
            PET_recon_x, PET_mu, PET_logvar, PET_z, \
            SNP_recon_x, SNP_mu, SNP_logvar, SNP_z, \
            PET_mu_rg, PET_logvar_rg, PET_z_rg, \
            SNP_mu_rg, SNP_logvar_rg, SNP_z_rg, PET_y, SNP_y, muti_y, muti_z_f, sMRI_AM, PET_AM = model(sMRI, PET, new_label_PET, SNP, new_label_SNP, device)


            z_f_all.append(np.squeeze(muti_z_f.cpu().numpy()))
            label_all.append(np.squeeze(label.cpu().numpy()))


z_f_all = np.array(z_f_all)
labels = np.array(label_all)


def plot_3d_tsne(data, labels, title):
    tsne = TSNE(n_components=3, random_state=0)
    z_tsne = tsne.fit_transform(data)


    tsne_df = pd.DataFrame(z_tsne, columns=['tSNE1', 'tSNE2', 'tSNE3'])
    tsne_df['Label'] = labels


    tsne_df.to_csv(f'./tSNE_results_with_labels.csv', index=False)


    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')


    colors = {
        0: 'royalblue',
        1: 'lightskyblue',
        2: 'green',
        3: 'pink',
        4: 'darkred'
    }

    label_order = [0, 1, 2, 3, 4]

    for label in label_order:
        ax.scatter(z_tsne[labels == label, 0], z_tsne[labels == label, 1], z_tsne[labels == label, 2],
                   color=colors[label], label=f"Label {label}", alpha=0.4, s=25)


    ax.set_xlabel('tSNE1', fontsize=12, labelpad=10)
    ax.set_ylabel('tSNE2', fontsize=12, labelpad=10)
    ax.set_zlabel('tSNE3', fontsize=12, labelpad=10)


    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')


    ax.set_box_aspect([1, 1, 1])


    ax.xaxis.set_tick_params(direction='out', width=1.5, pad=5)
    ax.yaxis.set_tick_params(direction='out', width=1.5, pad=5)
    ax.zaxis.set_tick_params(direction='out', width=1.5, pad=5)


    axis_limit = 18
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])


    ax.legend(title="Labels", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)


    ax.set_title(title, fontsize=14)


    plt.savefig(f'./t-SNE_visualization_of_z_f.png', bbox_inches='tight', dpi=600)


    plt.show()


plot_3d_tsne(z_f_all, labels, 't-SNE Visualization of z_f_all')



