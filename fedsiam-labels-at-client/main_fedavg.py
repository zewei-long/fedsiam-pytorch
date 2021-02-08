#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from data.sampling import sample, noniid_ssl
from data.randaugment import RandomTranslateWithReflect
from utils.options import args_parser
from models.Update import LocalUpdate_fedavg, DatasetSplit
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        trans_mnist_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist_test)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar_test)
    elif args.dataset == 'svhn':
        transform_svhn = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_svhn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset_train = datasets.SVHN('../data/svhn',  split='train', download=True, transform=transform_svhn)
        dataset_test = datasets.SVHN('../data/svhn',  split='test', download=True, transform=transform_svhn_test)
    else:
        exit('Error: unrecognized dataset')

    dataset_valid = dataset_test

    if args.iid == 'noniid_ssl' and args.dataset == 'cifar':
        dict_users, dict_users_labeled, pseudo_label = noniid_ssl(dataset_train, args.num_users, args.label_rate)
    else:
        dict_users, dict_users_labeled, pseudo_label = sample(dataset_train, args.num_users, args.label_rate, args.iid)


    if args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    print("\n Begin Train")

    net_glob.train()
    w_glob = net_glob.state_dict()

    w_best = copy.deepcopy(w_glob)
    best_loss_valid = 1e10
    loss_train = []


    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        net_glob.train()
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate_fedavg(
                args = args, 
                dataset = dataset_train, 
                idxs = dict_users[idx], 
                idxs_labeled = dict_users_labeled[idx], 
                pseudo_label = pseudo_label
                )
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w)) 
            loss_locals.append(copy.deepcopy(loss))

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_valid, loss_valid = test_img(net_glob, dataset_valid, args)
        if loss_valid <= best_loss_valid:
            best_loss_valid = loss_valid
            w_best = copy.deepcopy(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, acc_valid {:.2f}%'.format(iter, loss_avg, acc_valid))
        loss_train.append(loss_avg)

    print("\n Begin test")

    net_glob.load_state_dict(w_best)
    net_glob.eval()

    users_labeled=set()
    for i in range(len(dict_users_labeled)) :
        users_labeled = users_labeled | dict_users_labeled[i]
    users_unlabeled=set()
    for i in range(len(dict_users_labeled)) :
        users_unlabeled = users_unlabeled | (dict_users[i] - dict_users_labeled[i])
    dataset_train_labeled = DatasetSplit(dataset = dataset_train, idxs = users_labeled, pseudo_label = pseudo_label)


    acc_train_labeled, loss_train_test_labeled = test_img(net_glob, dataset_train_labeled, args)
    print("labeled Training accuracy: {:.2f}%".format(acc_train_labeled))    

    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}% \n\n".format(acc_test))
    
