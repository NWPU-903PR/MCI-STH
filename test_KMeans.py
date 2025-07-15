#To cluster and stage the MCI samples

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import warnings
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
from torch.autograd import Variable

from model import VAE_all
from utils.nii_read import load_nii_from_txt
from utils.csv_read import load_csv_from_txt
from utils.Data_label_preprocess_test import AD_MultiData

warnings.filterwarnings(action='ignore')
device = torch.device("cpu")  # Or "cuda" for GPU acceleration

# Paths
IMG_PATH_sMRI = 'data/mwp1_nii/'
IMG_PATH_PET = 'data/AV45-PET/'
dataset_SNP = pd.read_csv('data/Data_SNP.csv', index_col='ID')
dataset_PATH = "data/test_set.txt"

# Load data
nii_data_dict, PET_data_dict, PET_label_dict = load_nii_from_txt(dataset_PATH, IMG_PATH_sMRI, IMG_PATH_PET)
SNP_data_dict, SNP_label_dict = load_csv_from_txt(dataset_PATH, dataset_SNP)

# Prepare dataset
dset_test = AD_MultiData(dataset_PATH, nii_data_dict, PET_data_dict, PET_label_dict, SNP_data_dict, SNP_label_dict)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)

def calculate_rss(X, labels):
    rss = 0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        cluster_center = np.mean(cluster_points, axis=0)
        rss += np.sum((cluster_points - cluster_center)**2)
    return rss

# Initialize the model
model = VAE_all()
model.to(device)
n_clusters = 3  # Number of clusters

# Load the pretrained model
model.eval()
model_weight_path = "model/MCI_STH.pth"
model.load_state_dict(torch.load(model_weight_path))

# Variables for storing latent variables
z_f_all = []
label_all = []

# Extract latent variables
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

            # Create new label masks
            mask_PET = torch.zeros(label.shape)
            new_label_PET = mask_PET.to(device)
            mask_SNP = torch.zeros(label.shape)
            new_label_SNP = mask_SNP.to(device)

            # Forward pass
            sMRI_recon_x, sMRI_mu, sMRI_logvar, sMRI_z, \
            PET_recon_x, PET_mu, PET_logvar, PET_z, \
            SNP_recon_x, SNP_mu, SNP_logvar, SNP_z, \
            PET_mu_rg, PET_logvar_rg, PET_z_rg, \
            SNP_mu_rg, SNP_logvar_rg, SNP_z_rg, PET_y, SNP_y, muti_y, muti_z_f, sMRI_AM, PET_AM = model(
                sMRI, PET, new_label_PET, SNP, new_label_SNP, device)

            z_f_all.append(np.squeeze(muti_z_f.cpu().numpy()))
            label_all.append(np.squeeze(label.cpu().numpy()))

# Convert to numpy arrays
sample_ids = list(nii_data_dict.keys())
z_f_all = np.array(z_f_all)
label_all = np.array(label_all)

# Filter samples with label = 1 (assuming 1 represents MCI)
indices_label_MCI = np.where(label_all == 1)[0]
muti_z_f_label_MCI = z_f_all[indices_label_MCI]
sample_ids_label_MCI = [sample_ids[i] for i in indices_label_MCI]

# ============ z_f Cluster============
kmeans_z_f = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_z_f.fit(muti_z_f_label_MCI)
labels_z_f = kmeans_z_f.labels_

# Save clustering results for z_f to CSV
cluster_result_z_f_f = pd.DataFrame({'CombinedName': sample_ids_label_MCI, 'Cluster': labels_z_f + 1})
cluster_result_z_f_f.loc[len(cluster_result_z_f_f)] = ['Silhouette Coefficient', metrics.silhouette_score(muti_z_f_label_MCI, labels_z_f)]
cluster_result_z_f_f.loc[len(cluster_result_z_f_f)] = ['Calinski-Harabasz Index', metrics.calinski_harabasz_score(muti_z_f_label_MCI, labels_z_f)]
cluster_result_z_f_f.loc[len(cluster_result_z_f_f)] = ['Davies-Bouldin Index', metrics.davies_bouldin_score(muti_z_f_label_MCI, labels_z_f)]
cluster_result_z_f_f.loc[len(cluster_result_z_f_f)] = ['Residual Sum of Squares', calculate_rss(muti_z_f_label_MCI, labels_z_f)]
cluster_result_z_f_f.to_csv(f"./cluster_result.csv", index=False)