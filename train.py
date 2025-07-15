import torch
from torch.autograd import Variable

from loss_function import VAE_loss
from loss_function import VAE_loss_label
from loss_function import regression_loss


def train(model, train_loader, device, optimizer, classification_loss,  weight_ELBO = 1, weight_MSE = 1, weight_KL = 1,\
          weight_CMG = 1, weight_MMC = 1, weight_AC_PET = 1, weight_AC_SNP = 1):
    # main training loop
    train_loss_total = 0.0
    train_loss_ELBO = 0.0
    train_loss_CMG_PET = 0.0
    train_loss_CMG_SNP = 0.0

    train_loss_MMC = 0.0
    train_loss_AC_PET = 0.0
    train_loss_AC_SNP = 0.0


    model.train()
    for it, train_data in enumerate(train_loader):
        for data_dic in train_data:

            sMRI, PET, label_PET, SNP, label_SNP, label = Variable(data_dic['sMRI']).float().to(device),\
                                                          Variable(data_dic['PET']).float().to(device), Variable(data_dic['PET label']).float().to(device), \
                                                          Variable(data_dic['SNP']).float().to(device), Variable(data_dic['SNP label']).float().to(device), \
                                                          Variable(data_dic['label']).to(device)

            true_labels = label.data.cpu().numpy()
            # target should be LongTensor in loss function
            true_labels = Variable(torch.from_numpy(true_labels)).long()
            true_labels = true_labels .to(device)

            sMRI_recon_x, sMRI_mu, sMRI_logvar, sMRI_z,\
            PET_recon_x, PET_mu, PET_logvar, PET_z, \
            SNP_recon_x, SNP_mu, SNP_logvar, SNP_z, \
            PET_mu_rg, PET_logvar_rg, PET_z_rg, \
            SNP_mu_rg, SNP_logvar_rg, SNP_z_rg, PET_y, SNP_y, muti_y, muti_z_f, sMRI_AM, PET_AM = model(sMRI, PET, label_PET, SNP, label_SNP, device)


            loss_ELBO_sMRI = VAE_loss(sMRI, sMRI_recon_x, sMRI_mu, sMRI_logvar, weight_MSE, weight_KL)
            loss_ELBO_PET = VAE_loss_label(PET, PET_recon_x, PET_mu, PET_logvar, label_PET, weight_MSE, weight_KL)
            loss_ELBO_SNP = VAE_loss_label(SNP, SNP_recon_x, SNP_mu, SNP_logvar, label_SNP, weight_MSE, weight_KL)

            loss_ELBO = loss_ELBO_sMRI + loss_ELBO_PET + loss_ELBO_SNP
            train_loss_ELBO += loss_ELBO

            loss_rg_PET_mu = regression_loss(PET_mu, PET_mu_rg, label_PET)
            loss_rg_PET_logvar = regression_loss(PET_logvar, PET_logvar_rg, label_PET)
            loss_rg_SNP_mu = regression_loss(SNP_mu, SNP_mu_rg, label_SNP)
            loss_rg_SNP_logvar = regression_loss(SNP_logvar, SNP_logvar_rg, label_SNP)

            loss_CMG_PET = loss_rg_PET_mu + loss_rg_PET_logvar
            train_loss_CMG_PET += loss_CMG_PET
            loss_CMG_SNP = loss_rg_SNP_mu + loss_rg_SNP_logvar
            train_loss_CMG_SNP += loss_CMG_SNP

            loss_MMC = classification_loss(muti_y, true_labels)
            train_loss_MMC += loss_MMC

            loss_AC_PET = classification_loss(PET_y, true_labels)
            train_loss_AC_PET += loss_AC_PET

            loss_AC_SNP = classification_loss(SNP_y, true_labels)
            train_loss_AC_SNP += loss_AC_SNP

            loss_total = weight_ELBO * loss_ELBO + weight_CMG * (loss_CMG_PET + loss_CMG_SNP) \
                         + weight_MMC * loss_MMC\
                         + weight_AC_PET * loss_AC_PET\
                         + weight_AC_SNP * loss_AC_SNP


            train_loss_total += loss_total

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

    return train_loss_total, train_loss_ELBO, \
           train_loss_CMG_PET, train_loss_CMG_SNP, \
           train_loss_MMC, train_loss_AC_PET, train_loss_AC_SNP