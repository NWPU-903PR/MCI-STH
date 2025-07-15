import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '0'
import argparse
import logging
import torch
import warnings
import pandas as pd
from torch.utils.data import DataLoader

from utils.make_set import make_set
from model import VAE_all
from utils.nii_read import load_nii_from_txt
from utils.csv_read import load_csv_from_txt
from utils.Data_label_preprocess import AD_MultiData
from train import train
from val import validate


print(torch.cuda.device_count())
warnings.filterwarnings(action='ignore')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for Python")

parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="'.pth",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=200,
                    help="Epochs through the data. (default=2000)")
parser.add_argument("--learning_rate", "-lr", default=1e-4,type=float,
                    help="Learning rate of the optimization. (default=1e-4)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=5, type=int,
                    help="Batch size for training. (default=10)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


# feel free to add more arguments as you need

def main(options):

    IMG_PATH_sMRI = 'data/mwp1_nii/'
    IMG_PATH_PET = 'data/AV45-PET/'
    dataset_SNP = pd.read_csv('data/Data_SNP.csv', index_col='ID')
    dataset_PATH = "data/train+val_set.txt"
    sMRI_data_dict, PET_data_dict, PET_label_dict = load_nii_from_txt(dataset_PATH, IMG_PATH_sMRI,IMG_PATH_PET)
    SNP_data_dict, SNP_label_dict = load_csv_from_txt(dataset_PATH, dataset_SNP)

    weight_ELBO = 10
    weight_MSE = 1
    weight_KL = 1e-6
    weight_CMG = 1
    weight_MMC = 10
    weight_AC_PET = 10
    weight_AC_SNP = 10


    for m in range(10):
        logging.info("At {0}-th cross.".format(m))
        dataset_name = "data/train+val_set.txt"
        make_set(m, dataset_name, 0.8)
        TRAINING_PATH = "data/train_cross{0}.txt".format(m)
        VALING_PATH = "data/val_cross{0}.txt".format(m)
        dset_train = AD_MultiData(TRAINING_PATH, sMRI_data_dict, PET_data_dict, PET_label_dict, SNP_data_dict, SNP_label_dict)
        dset_val = AD_MultiData(VALING_PATH, sMRI_data_dict,PET_data_dict, PET_label_dict, SNP_data_dict, SNP_label_dict)


        if options.load is None:
            train_loader = DataLoader(dset_train, batch_size=options.batch_size, shuffle=True, num_workers=1, drop_last=True)

        else:
            train_loader = DataLoader(dset_train, batch_size=options.batch_size, shuffle=False, num_workers=1, drop_last=True)

        val_loader = DataLoader(dset_val, batch_size=options.batch_size, shuffle=False, num_workers=1, drop_last=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = VAE_all()
        model.to(device)

        lr = options.learning_rate
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

        best_loss = 1e9
        classification_loss = torch.nn.NLLLoss()

        for epoch_i in range(options.epochs):
            logging.info("At {0}-th epoch.".format(epoch_i))

            train_loss_total, train_loss_ELBO, \
            train_loss_CMG_PET, train_loss_CMG_SNP, \
            train_loss_MMC, train_loss_AC_PET, train_loss_AC_SNP = train(model, train_loader, device, optimizer,\
                                                                                        classification_loss, weight_ELBO, weight_MSE,\
                                                                                        weight_KL, weight_CMG, weight_MMC,\
                                                                                        weight_AC_PET, weight_AC_SNP)

            train_avg_loss_total = train_loss_total / (len(dset_train) / options.batch_size)
            train_avg_loss_ELBO = train_loss_ELBO / (len(dset_train) / options.batch_size)
            train_avg_loss_CMG_PET = train_loss_CMG_PET / (len(dset_train) / options.batch_size)
            train_avg_loss_CMG_SNP = train_loss_CMG_SNP / (len(dset_train) / options.batch_size)
            train_avg_loss_MMC = train_loss_MMC / (len(dset_train) / options.batch_size)
            train_avg_loss_AC_PET = train_loss_AC_PET / (len(dset_train) / options.batch_size)
            train_avg_loss_AC_SNP = train_loss_AC_SNP / (len(dset_train) / options.batch_size)


            logging.info("Average training loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_total.item(), epoch_i))
            logging.info("Average training ELBO loss is {0:.12f} at the end of epoch {1}".format( train_avg_loss_ELBO.item(), epoch_i))
            logging.info("Average training PET CMG loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_CMG_PET.item(), epoch_i))
            logging.info("Average training SNP CMG loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_CMG_SNP.item(), epoch_i))
            logging.info("Average training MMC loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_MMC.item(), epoch_i))
            logging.info("Average training PET AC loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_AC_PET.item(), epoch_i))
            logging.info("Average training SNP AC loss is {0:.12f} at the end of epoch {1}".format(train_avg_loss_AC_SNP.item(), epoch_i))


            # write the training loss acu to file

            val_loss_total, val_loss_ELBO, \
            val_loss_CMG_PET, val_loss_CMG_SNP, \
            val_loss_MMC, val_loss_AC_PET, val_loss_AC_SNP = validate(model, val_loader, device, classification_loss, weight_ELBO,\
                                                                                     weight_MSE, weight_KL, weight_CMG, weight_MMC,\
                                                                                     weight_AC_PET, weight_AC_SNP)

            val_avg_loss_total = val_loss_total / (len(dset_val) / options.batch_size)
            val_avg_loss_ELBO = val_loss_ELBO / (len(dset_val) / options.batch_size)
            val_avg_loss_CMG_PET = val_loss_CMG_PET / (len(dset_val) / options.batch_size)
            val_avg_loss_CMG_SNP = val_loss_CMG_SNP / (len(dset_val) / options.batch_size)
            val_avg_loss_MMC = val_loss_MMC / (len(dset_val) / options.batch_size)
            val_avg_loss_AC_PET = val_loss_AC_PET / (len(dset_val) / options.batch_size)
            val_avg_loss_AC_SNP = val_loss_AC_SNP / (len(dset_val) / options.batch_size)


            scheduler.step(val_avg_loss_total)
            logging.info("Average valing loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_total.item(), epoch_i))
            logging.info("Average valing ELBO loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_ELBO.item(), epoch_i))
            logging.info("Average valing PET CMG loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_CMG_PET.item(), epoch_i))
            logging.info("Average valing SNP CMG loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_CMG_SNP.item(), epoch_i))
            logging.info("Average valing MMC loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_MMC.item(), epoch_i))
            logging.info("Average valing PET AC loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_AC_PET.item(), epoch_i))
            logging.info("Average valing SNP AC loss is {0:.12f} at the end of epoch {1}".format(val_avg_loss_AC_SNP.item(), epoch_i))


            # write validation accuracy to file

            if val_avg_loss_total < best_loss:
                best_loss = val_avg_loss_total
                best_epoch = epoch_i
                print(best_epoch)
                torch.save(model.state_dict(),open("/MCI_STH.pth".format(m), 'wb') )





if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format (parser.parse_known_args()[1]))
    main(options)
