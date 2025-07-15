# Uncovering Spatiotemporal Heterogeneity in Mild Cognitive Impairment via Incomplete Multi-Modal Data Integration

MCI-STH is a deep learning framework designed to investigate the spatiotemporal heterogeneity of Mild Cognitive Impairment (MCI). This model utilizes sMRI, AV45-PET, and SNP data to construct the trajectory of Alzheimer’s disease (AD) progression.

## Requirements
The project is written in Python 3.9 and all experiments were conducted on Linux OS with 24G×4 NVIDIA RTX 3090 GPU, 40×2.4GHZ Intel Xeon CPU, and 128GB RAM. For the faster training process, training on a GPU is necessary, but a standard computer without GPU also works (consuming much more training time). 

All implementations of MCI-STH and the VAE-based baselines were based on PyTorch. MCI-STH requires the following dependencies:

- python == 3.9
- numpy == 1.25.0
- pandas == 2.0.3
- nibable == 5.1.0
- pytorch == 2.0.1

## Reproducibility

### 1. Train

- `main.py ` is used to train MCI-STH, using only AD and NC samples for training.

### 2. Test 

-  `test_KMeans.py` is used to perform clustering on MCI samples and compute clustering evaluation metrics.
- `test_MCI_stage_label.py` is used to refine the clustering labels based on the severity of dementia in patients.
- `test_result_NSD.py` is used to calculate the number of neuropsychological tests that can distinguish between the MCI stages identified by MCI-STH.

### 3. Visualization

- `visual_Z_f_t-SNE.py` is used to perform dimensionality reduction on the multimodal fused features and visualize the results.
- `visual_trajectory.py` is used to generate 3D visualizations of the AD progression trajectory in the latent space from different perspectives.




