#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import matplotlib
matplotlib.use('Agg')
import copy
from torchvision import datasets, transforms
import torch

from data.sampling import sample, noniid_ssl
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from data.randaugment import RandomTranslateWithReflect

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        trans_mnist_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist_test)
        dataset_train_ema = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist_test)
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_train_ema = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar_test)

    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_svhn_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_train = datasets.SVHN('../data/svhn',  split='train', download=True, transform=trans_svhn_test)
        dataset_train_ema = datasets.SVHN('../data/svhn',  split='train', download=True, transform=trans_svhn_test)
        dataset_test = datasets.SVHN('../data/svhn',  split='test', download=True, transform=trans_svhn_test)

    else:
        exit('Error: unrecognized dataset')

    dataset_valid = dataset_test

    if args.iid == 'noniid_ssl' and args.dataset == 'cifar':
        dict_users, dict_users_labeled, pseudo_label = noniid_ssl(dataset_train, args.num_users, args.label_rate)
    else:
        dict_users, dict_users_labeled, pseudo_label = sample(dataset_train, args.num_users, args.label_rate, args.iid)


    if args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_ema_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        net_ema_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
        net_ema_glob = CNNCifar(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()
    net_ema_glob.train()

    w_glob = net_glob.state_dict()
    w_ema_glob = net_ema_glob.state_dict()

    w_best = copy.deepcopy(w_glob)
    w_ema_best = copy.deepcopy(w_ema_glob)
    best_loss_valid = 100000000000
    best_ema_loss_valid = 100000000000

    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    dict_userepoch={i:0 for i in range(100)}

    for iter in range(args.epochs):

        net_glob.train()
        net_ema_glob.train()

        w_locals, w_ema_locals, loss_locals, loss_consistent_locals = [], [], [], []
        m = max(int(args.frac * args.num_users), 1)#choice trained users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            dict_userepoch[idx] = dict_userepoch[idx] + 1
            local = LocalUpdate(args=args, dataset=dataset_train, dataset_ema=dataset_train_ema, 
                                idxs=dict_users[idx], idxs_labeled=dict_users_labeled[idx], 
                                pseudo_label=pseudo_label)
            
            w, w_ema ,loss, loss_consistent = local.train(
                    net=copy.deepcopy(net_glob).to(args.device),
                    net_ema=copy.deepcopy(net_ema_glob).to(args.device),
                    args=args, iter_glob=iter+1, user_epoch=dict_userepoch[idx])
            
            w_locals.append(copy.deepcopy(w))
            w_ema_locals.append(copy.deepcopy(w_ema))
            loss_locals.append(copy.deepcopy(loss))
            loss_consistent_locals.append(copy.deepcopy(loss_consistent))

        w_glob = FedAvg(w_locals)
        w_ema_glob = FedAvg(w_ema_locals)

        net_glob.load_state_dict(w_glob)
        net_ema_glob.load_state_dict(w_ema_glob)

        net_glob.eval()
        net_ema_glob.eval()
        acc_valid, loss_valid = test_img(net_glob, dataset_valid, args)
        acc_ema_valid, loss_ema_valid = test_img(net_ema_glob, dataset_valid, args)
        if loss_valid <= best_loss_valid:
            best_loss_valid = loss_valid
            w_best = copy.deepcopy(w_glob)
        if loss_ema_valid <= best_ema_loss_valid:
            best_ema_loss_valid = loss_ema_valid
            w_ema_best = copy.deepcopy(w_ema_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_consistent_avg = sum(loss_consistent_locals) / len(loss_consistent_locals)
        print('Round {:3d}, Average loss {:.3f}, Average loss_consist {:.3f}, acc_valid {:.2f}%, acc_ema_valid {:.2f}%'
            .format(iter, loss_avg, loss_consistent_avg, acc_valid, acc_ema_valid))
        loss_train.append(loss_avg)



    net_glob.load_state_dict(w_best)
    net_ema_glob.load_state_dict(w_ema_best)

    net_glob.eval()
    net_ema_glob.eval()
    acc_train, loss_train_f = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("\ns:Training accuracy: {:.4f}".format(acc_train))
    print("s:Testing accuracy: {:.4f}".format(acc_test))

    acc_train, loss_train_f = test_img(net_ema_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_ema_glob, dataset_test, args)
    print("t:Training accuracy: {:.4f}".format(acc_train))
    print("t:Testing accuracy: {:.4f}".format(acc_test))