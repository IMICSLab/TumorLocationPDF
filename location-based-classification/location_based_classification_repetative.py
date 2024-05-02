#!/usr/bin/env python3
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
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import shutil
#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from imicsDataset import iMICSDataset
from imicsDataset import load_object
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy import percentile
import matplotlib.pyplot as plt
import scipy.stats as st
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

"""
100 runs:

"""
def five_number_summary(lst):
    quartiles = percentile(lst, [25,50,75])
    data_min, data_max = min(lst), max(lst)
    return data_min, quartiles[0], quartiles[1], quartiles[2], data_max

def loc_clf_experiment(N_repeats, ROIs, labels, pIDs):
    Perfs = []
    for exp_i in range(N_repeats):
        print("working on experimnet number", exp_i)
        N = len(labels) #Number of patients
        inds = [i for i in range(N)]
        ind_dev, ind_test, y_dev, y_test = train_test_split(inds, labels, test_size=0.2, random_state=exp_i)
    
        train_dset = iMICSDataset(ROIs, labels, pIDs, ind_dev, shuffle=True) #we could turn off shuffling because train_test_split shuffles the inds
        test_dset = iMICSDataset(ROIs, labels, pIDs, ind_test, shuffle=False)
    
        sum_masks_lgg = np.zeros(ref_size)
        sum_masks_hgg = np.zeros(ref_size)
        for i in range(len(train_dset)):
            if train_dset.labels[i] == 0:
                sum_masks_lgg += train_dset.ROIs[i]
            elif train_dset.labels[i] == 1:
                sum_masks_hgg += train_dset.ROIs[i]
            else:
                print("Errrrrrrrrrrrrror!")
        pdf_lgg_loc = sum_masks_lgg/np.sum(sum_masks_lgg)
        pdf_hgg_loc = sum_masks_hgg/np.sum(sum_masks_hgg)
    
        predictions = []
        for i in range(len(test_dset)):
            patient_p_lgg = np.sum(np.multiply(test_dset.ROIs[i], pdf_lgg_loc))
            patient_p_hgg = np.sum(np.multiply(test_dset.ROIs[i], pdf_hgg_loc))
            if patient_p_hgg>patient_p_lgg:
                predictions.append(min(0.5+patient_p_hgg, 1))
                #predictions.append(1)
            else:
                predictions.append(max(0.5-patient_p_lgg, 0))
                #predictions.append(0)
        Perfs.append(roc_auc_score(test_dset.labels, predictions))
        print("test AUC: {}".format(roc_auc_score(test_dset.labels, predictions)))
        del train_dset, test_dset
    return Perfs


if __name__ == "__main__":
    ROIs = load_object("./binMasks.p")
    labels = load_object("./binlabels.p")
    pIDs = load_object("./binpIDs.p")
    ref_size = (240, 240, 155)
    AUCs = loc_clf_experiment(100, ROIs, labels, pIDs)
    print(np.mean(AUCs))
    print(np.std(AUCs))
    print(five_number_summary(AUCs))
    
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    plt.boxplot(AUCs, labels=[""], meanprops=meanpointprops, meanline=False,
                      showmeans=True)
    plt.gca().yaxis.grid(True, color='moccasin')
    plt.ylabel("AUROC")
    
    st.t.interval(alpha=0.95, df=len(AUCs)-1, loc=np.mean(AUCs), scale=st.sem(AUCs))

