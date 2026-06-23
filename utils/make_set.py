import random
from data_split import data_split

def make_set(m, dataset_name, random_ratio):

    df = open(dataset_name)
    lines = df.readlines()
    AD = []
    NC = []
    for idx in range(0, len(lines)):
        lst = lines[idx].split()
        if lst[1] == '3':
            AD.append(lst)
        if lst[1] == '1':
            NC.append(lst)
#
    AD_train, AD_val = data_split(AD, ratio=random_ratio, shuffle=True)
    NC_train, NC_val = data_split(NC, ratio=random_ratio, shuffle=True)

    train_data = AD_train + NC_train
    random.shuffle(train_data)
    val_data = AD_val + NC_val
    random.shuffle(val_data)


    train_ft_f = open("data/train_cross{0}.txt".format(m), "w")
    val_ft_f = open("data/val_cross{0}.txt".format(m), "w")

    for train_name, train_label in train_data:
        X_train, Y_train = train_name, train_label
        train_ft_f.write(X_train + '\t' + Y_train + '\n')

    for val_name, val_label in val_data:
        X_val, Y_val = val_name, val_label
        val_ft_f.write(X_val + '\t' + Y_val + '\n')

    train_ft_f.close()
    val_ft_f.close()




