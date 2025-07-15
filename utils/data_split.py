import random


def data_split(full_list, ratio, shuffle=False):

    n_total = len(full_list)
    offset = int(n_total * ratio)

    if n_total == 0 or offset < 1:

        return [], full_list
    if shuffle:

        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]

    return sublist_1, sublist_2
