# -*- coding: utf-8 -*-
"""
@author: Ernest Namdar (ernest.namdar@utoronto.ca)
"""
# importing the required libraries#############################################
# >>>>Seeding function libraries
import numpy as np
import torch
import random
# >>>>other required libs
import pickle
from torch.utils.data import Dataset


# Seeding######################################################################
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(0, True)
###############################################################################


class iMICSDataset(Dataset):
    def __init__(self, ROIs, labels, pIDs, inds, shuffle):
        self.len = len(inds)
        self.ROIs = [ROIs[i] for i in inds]
        self.labels = [labels[i] for i in inds]
        self.pIDs = [pIDs[i] for i in inds]

        if shuffle is True:
            p = np.random.permutation(self.len)
            self.ROIs = [self.ROIs[i] for i in p]
            self.labels = [self.labels[i] for i in p]
            self.pIDs = [self.pIDs[i] for i in p]
        self.ROIs = np.stack(self.ROIs, axis=0)

    def __len__(self):
        return (self.len)

    def __getitem__(self, idx):
        roi = self.ROIs[idx, :, :, :]
        lbl = self.labels[idx]
        pid = self.pIDs[idx]
        sample = {'ROI': roi, 'label': lbl, 'pID': pid}
        return sample
    def positive_ratio(self):
        count = 0
        for lb in self.labels:
            if lb==1:
                count += 1
        return count/len(self)
    def negative_ratio(self):
        return 1-self.positive_ratio()


def load_object(filename):
  # opening a file in read, binary form
  file = open(filename, 'rb')

  ob = pickle.load(file)

  # close the file
  file.close()
  return ob


if __name__ == '__main__':
    ROIs = load_object("./Masks.p")
    labels = load_object("./labels.p")
    pIDs = load_object("./pIDs.p")

    inds = [3,4,5,7,10]
    dset = iMICSDataset(ROIs, labels, pIDs, inds, shuffle=True)
