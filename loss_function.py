import torch
import torch.nn.functional as F


def VAE_loss(y, y_hat, mean, logvar, mse_weight, kl_weight):

    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)

    loss = recons_loss * mse_weight + kl_loss * kl_weight

    return loss


def VAE_loss_label(y, y_hat, mean, logvar, label, mse_weight, kl_weight):

    mask = (label != 0).view(-1)

    y_valid = y[mask]
    y_hat_valid = y_hat[mask]
    mean_valid = mean[mask]
    logvar_valid = logvar[mask]

    if mask.sum() == 0:
        return torch.tensor(0.0, device=y.device), torch.tensor(0.0, device=y.device), torch.tensor(0.0,
                                                                                                    device=y.device)

    recons_loss = F.mse_loss(y_hat_valid, y_valid)

    kl_loss = -0.5 * torch.sum(1 + logvar_valid - mean_valid.pow(2) - logvar_valid.exp(), dim=1)
    kl_loss = kl_loss.mean() 

    loss = recons_loss * mse_weight + kl_loss * kl_weight

    return loss


def regression_loss(y, y_hat, label):

    y_1 = (label != 0).nonzero().squeeze()
    y = y[y_1]
    y_hat =y_hat[y_1]

    regression_loss = F.mse_loss(y_hat, y, reduction='mean')

    return regression_loss





