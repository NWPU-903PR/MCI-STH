import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))

        return x


class encoder_sMRI(nn.Module):
    def __init__(self):
        super(encoder_sMRI, self).__init__()
        self.attention = Attention(kernel_size=7)

        self.conv1 = nn.Conv3d(1, 15, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(15, 25, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(25, 50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(50, 50, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(1, 0, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(50 * 7 * 8 * 7, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        attention = self.attention(x)
        x_attention = x * attention

        x = self.relu(self.conv1(x_attention))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(-1, 50 * 7 * 8 * 7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        z_mean, z_logvar = torch.chunk(x, 2, dim=1)

        return z_mean, z_logvar, attention


class decoder_sMRI(nn.Module):
    def __init__(self):
        super(decoder_sMRI, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 50*7*8*7)
        self.conv1 = nn.ConvTranspose3d(50, 50, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose3d(50, 25, kernel_size=3, stride=2, padding=1, output_padding=(0, 1, 0))
        self.conv3 = nn.ConvTranspose3d(25, 15, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose3d(15, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = x.view(-1, 50, 7, 8, 7)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x


class VAE_sMRI(nn.Module):
    def __init__(self):
        super(VAE_sMRI, self).__init__()
        self.encoder_sMRI = encoder_sMRI()
        self.decoder_sMRI = decoder_sMRI()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)

        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        mu, logvar, attention = self.encoder_sMRI(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder_sMRI(z)

        return recon_x, mu, logvar, z, attention


class encoder_PET(nn.Module):
    def __init__(self):
        super(encoder_PET, self).__init__()
        self.attention = Attention(kernel_size=7)

        self.conv1 = nn.Conv3d(1, 15, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(15, 25, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(25, 50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(50, 50, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(0, 1, 0))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(1, 0, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(50 * 6 * 7 * 6, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = x.unsqueeze(0)
        attention = self.attention(x_1)
        x_attention = x * attention.squeeze(0)

        x = self.relu(self.conv1(x_attention))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(-1, 50 * 6 * 7 * 6)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        z_mean, z_logvar = torch.chunk(x, 2, dim = -1)

        return z_mean, z_logvar, attention


class decoder_PET(nn.Module):
    def __init__(self):
        super(decoder_PET, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 50 * 6 * 7 * 6)
        self.conv1 = nn.ConvTranspose3d(50, 50, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(50, 25, kernel_size=3, stride=2, padding=1, output_padding=(0, 1, 0))
        self.conv3 = nn.ConvTranspose3d(25, 15, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 1))
        self.conv4 = nn.ConvTranspose3d(15, 1, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = x.view(-1, 50, 6, 7, 6)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x


class VAE_PET(nn.Module):
    def __init__(self):
        super(VAE_PET, self).__init__()
        self.encoder_PET = encoder_PET()
        self.decoder_PET = decoder_PET()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)

        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        mu, logvar, attention = self.encoder_PET(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder_PET(z)

        return recon_x, mu.squeeze(0), logvar.squeeze(0), z.squeeze(0), attention.squeeze(0)


class encoder_SNP(nn.Module):
    def __init__(self):
        super(encoder_SNP, self).__init__()
        self.fc1 = nn.Linear(1549, 256)
        self.fc2 = nn.Linear(256, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z_mean, z_logvar = torch.chunk(x, 2, dim = 0)

        return z_mean, z_logvar


class decoder_SNP(nn.Module):
    def __init__(self):
        super(decoder_SNP, self).__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 1549)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))

        return x


class VAE_SNP(nn.Module):
    def __init__(self):
        super(VAE_SNP, self).__init__()
        self.encoder_SNP = encoder_SNP()
        self.decoder_SNP = decoder_SNP()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)

        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        mu, logvar = self.encoder_SNP(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder_SNP(z)

        return recon_x, mu, logvar, z


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1_mu = nn.Linear(input_size, 16)
        self.fc2_mu = nn.Linear(16, output_size)
        self.fc1_logvar = nn.Linear(input_size, 16)
        self.fc2_logvar = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)

        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x1_mu, x1_logvar):
        x2_mu = self.relu(self.fc1_mu(x1_mu))
        x2_mu = self.relu(self.fc2_mu(x2_mu))

        x2_logvar = self.relu(self.fc1_logvar(x1_logvar))
        x2_logvar = self.relu(self.fc2_logvar(x2_logvar))

        z = self.reparameterise(x2_mu, x2_logvar)

        return x2_mu, x2_logvar, z


class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.BN = nn.BatchNorm1d(input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.logsoftmax =nn.LogSoftmax()


    def forward(self, z):

        z = self.BN(z)
        z_f = self.fc1(z)
        z_f_1 = self.relu(z_f)
        z_f_1 = self.dropout(z_f_1)
        z_f_1 = self.fc2(z_f_1)
        y = self.logsoftmax(z_f_1)

        return z_f, y



class VAE_all(nn.Module):
    def __init__(self, input_size=16, hidden_size=8, all_input_size=48, all_hidden_size=8, num_classes=2):
        super(VAE_all, self).__init__()
        self.VAE_all_sMRI = VAE_sMRI()
        self.VAE_all_PET = VAE_PET()
        self.VAE_all_SNP = VAE_SNP()
        self.RegressionModel_sMRI_PET = RegressionModel(input_size, input_size)
        self.RegressionModel_sMRI_SNP = RegressionModel(input_size, input_size)
        self.ClassificationModel_PET = ClassificationModel(input_size, hidden_size, num_classes)
        self.ClassificationModel_SNP = ClassificationModel(input_size, hidden_size, num_classes)
        self.ClassificationModel_all = ClassificationModel(all_input_size, all_hidden_size, num_classes)

    def forward(self, x1, x2, x2_label, x3, x3_label, device):

        PET_recon_x = [None] * len(x2)
        PET_mu = [None] * len(x2)
        PET_logvar = [None] * len(x2)
        PET_z = [None] * len(x2)
        PET_mu_rg = [None] * len(x2)
        PET_logvar_rg = [None] * len(x2)
        PET_z_rg = [None] * len(x2)
        PET_AM = [None] * len(x2)

        SNP_recon_x = [None] * len(x3)
        SNP_mu = [None] * len(x3)
        SNP_logvar = [None] * len(x3)
        SNP_z = [None] * len(x3)
        SNP_mu_rg = [None] * len(x3)
        SNP_logvar_rg = [None] * len(x3)
        SNP_z_rg = [None] * len(x3)

        sMRI_recon_x, sMRI_mu, sMRI_logvar, sMRI_z, sMRI_AM = self.VAE_all_sMRI(x1)

        for i in range(x2_label.size(0)):

            if x2_label[i] == 1:

                PET_recon_x[i], PET_mu[i], PET_logvar[i], PET_z[i], PET_AM[i] = self.VAE_all_PET(x2[i])
                PET_mu_rg[i], PET_logvar_rg[i], PET_z_rg[i] = self.RegressionModel_sMRI_PET(sMRI_mu[i], sMRI_logvar[i])

            else:

                PET_mu[i], PET_logvar[i], PET_z[i] = self.RegressionModel_sMRI_PET(sMRI_mu[i], sMRI_logvar[i])
                PET_recon_x[i] = self.VAE_all_PET.decoder_PET(PET_z[i])
                PET_mu_rg[i] = torch.zeros(16).to(device)
                PET_logvar_rg[i] = torch.zeros(16).to(device)
                PET_z_rg[i] = torch.zeros(16).to(device)
                PET_AM[i] = torch.zeros((1, 91, 109, 91)).to(device)

        for i in range(x3_label.size(0)):

            if x3_label[i] ==1:

                SNP_recon_x[i], SNP_mu[i], SNP_logvar[i], SNP_z[i] = self.VAE_all_SNP(x3[i])
                SNP_mu_rg[i], SNP_logvar_rg[i], SNP_z_rg[i] = self.RegressionModel_sMRI_SNP(SNP_mu[i], SNP_logvar[i])

            else:

                SNP_mu[i], SNP_logvar[i], SNP_z[i] = self.RegressionModel_sMRI_SNP(sMRI_mu[i], sMRI_logvar[i])
                SNP_recon_x[i] = self.VAE_all_SNP.decoder_SNP(SNP_z[i])
                SNP_mu_rg[i] = torch.zeros(16).to(device)
                SNP_logvar_rg[i] = torch.zeros(16).to(device)
                SNP_z_rg[i]= torch.zeros(16).to(device)


        PET_recon_x = torch.stack(PET_recon_x)
        PET_mu = torch.stack(PET_mu)
        PET_logvar = torch.stack(PET_logvar)
        PET_z = torch.stack(PET_z)
        PET_mu_rg = torch.stack(PET_mu_rg)
        PET_logvar_rg = torch.stack(PET_logvar_rg)
        PET_z_rg = torch.stack(PET_z_rg)
        PET_AM = torch.stack(PET_AM)

        SNP_recon_x = torch.stack(SNP_recon_x)
        SNP_mu = torch.stack(SNP_mu)
        SNP_logvar = torch.stack(SNP_logvar)
        SNP_z = torch.stack(SNP_z)
        SNP_mu_rg = torch.stack(SNP_mu_rg)
        SNP_logvar_rg = torch.stack(SNP_logvar_rg)
        SNP_z_rg = torch.stack(SNP_z_rg)

        PET_z_f, PET_y = self.ClassificationModel_PET(PET_z)
        SNP_z_f, SNP_y = self.ClassificationModel_SNP(SNP_z)
        muti_z = torch.cat( (sMRI_z, PET_z, SNP_z), 1)
        muti_z_f, muti_y = self.ClassificationModel_all(muti_z)

        return sMRI_recon_x, sMRI_mu, sMRI_logvar, sMRI_z, \
               PET_recon_x, PET_mu, PET_logvar, PET_z, \
               SNP_recon_x, SNP_mu, SNP_logvar, SNP_z, \
               PET_mu_rg, PET_logvar_rg, PET_z_rg, \
               SNP_mu_rg, SNP_logvar_rg, SNP_z_rg, PET_y, SNP_y, muti_y, muti_z_f, sMRI_AM, PET_AM



