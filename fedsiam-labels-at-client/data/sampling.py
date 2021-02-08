#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms

def noniid_ssl(dataset, num_users, label_rate):
    num_items = int(len(dataset)/num_users)
    dict_users, dict_users_labeled = {}, {}
    pseduo_label, all_idxs = [i for i in range(len(dataset))] ,[i for i in range(len(dataset))]
    label_rate_dict = {}
    for i in range(num_users):
        if i< 100 * label_rate:
            label_rate_dict[i] = (105 * label_rate -5) / (100 * label_rate) 
        else:
            label_rate_dict[i] = 0.05
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(num_items * label_rate_dict[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        for idxs in list(dict_users[i] - dict_users_labeled[i]):
            pseduo_label[idxs] = -1
        for idxs in list(dict_users_labeled[i]):
            pseduo_label[idxs] = dataset[idxs][1]
    return dict_users, dict_users_labeled, pseduo_label

def sample(dataset, num_users, label_rate, iid_if):
    num_items = int(len(dataset)/num_users)
    dict_users, dict_users_labeled = {}, {}
    pseduo_label, all_idxs = [i for i in range(len(dataset))] ,[i for i in range(len(dataset))]
    dict_idxs_class = {i : set() for i in range(10)}
    for idxs in range(len(dataset)):
        label_class = dataset[idxs][1]
        dict_idxs_class[label_class] = dict_idxs_class[label_class] | set([idxs])
        pseduo_label[idxs] = dataset[idxs][1]
    for i in range(10):
        dict_idxs_class[i] = list(dict_idxs_class[i])
    label_rate_dict = {}
    if iid_if == 'noniid_ssl':
        for i in range(num_users):
            if i< 100 * label_rate:
                label_rate_dict[i] = (105 * label_rate -5) / (100 * label_rate) 
            else:
                label_rate_dict[i] = 0.05
    else:
        for i in range(num_users):
            label_rate_dict[i] = label_rate

    for i in range(num_users):
        dict_users[i], dict_users_labeled[i], dict_idxs_class = get_dict(dict_idxs_class, num_users, iid_if, label_rate_dict[i], i)

        for idxs in list(dict_users[i] - dict_users_labeled[i]):
            pseduo_label[idxs] = -1
    if iid_if == 'noniid_improve':
        all_unlabeled_data = set()
        num_items_unlabeled = {}
        for i in range(num_users):
            all_unlabeled_data = all_unlabeled_data | (dict_users[i] - dict_users_labeled[i] )
            num_items_unlabeled[i] = len( (dict_users[i] - dict_users_labeled[i]) )
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(list(all_unlabeled_data), num_items_unlabeled[i], replace=False))
            all_unlabeled_data = list(set(all_unlabeled_data) - dict_users[i])
            dict_users[i] = dict_users[i] | dict_users_labeled[i]
    return dict_users, dict_users_labeled, pseduo_label

def get_dict(dict_idxs_class, num_users, iid_if, label_rate, i):

    dict_idxs = set()
    dict_idxs_labeled = set()
    if iid_if == 'iid' or iid_if == 'noniid_ssl':
        for key in dict_idxs_class.keys():
            idxs_class = dict_idxs_class[key][0 : int( len(dict_idxs_class[key]) / (num_users-i) ) ]
            idxs_class_labeled = idxs_class[0:round(len(idxs_class)*label_rate)]
            dict_idxs = dict_idxs | set(idxs_class)
            dict_idxs_labeled = dict_idxs_labeled | set(idxs_class_labeled)
            dict_idxs_class[key] = list(set(dict_idxs_class[key]) - set(idxs_class))
    elif iid_if == 'noniid_tradition' or iid_if == 'noniid_improve':
        x, y = i//10, i%10
        if x < y:
            rep_x, rep_y = x+y+1, x
        elif x == y:
            rep_x, rep_y = x+y, x+y+1
        else:
            rep_x, rep_y = x+y, x+9
        idxs_class = dict_idxs_class[x][0 : int( len(dict_idxs_class[x]) / (20 - rep_x)) ]
        idxs_class_labeled = idxs_class[0 : round(len(idxs_class)*label_rate)]
        dict_idxs = dict_idxs | set(idxs_class)
        dict_idxs_labeled = dict_idxs_labeled | set(idxs_class_labeled)
        dict_idxs_class[x] = list(set(dict_idxs_class[x]) - set(idxs_class))
        idxs_class = dict_idxs_class[y][0 : int( len(dict_idxs_class[y]) / (20 - rep_y)) ]
        idxs_class_labeled = idxs_class[0 : round(len(idxs_class)*label_rate)]
        dict_idxs = dict_idxs | set(idxs_class)
        dict_idxs_labeled = dict_idxs_labeled | set(idxs_class_labeled)
        dict_idxs_class[y] = list(set(dict_idxs_class[y]) - set(idxs_class))
    return dict_idxs, dict_idxs_labeled, dict_idxs_class

