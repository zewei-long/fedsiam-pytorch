#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
from torch.utils.data.sampler import Sampler
import re
import argparse
import os
import shutil
import time
import math
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from utils.losses import softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss
from utils.ramps import sigmoid_rampup, linear_rampup, cosine_rampdown, sigmoid_rampup2
from models.Nets import mnist_add, cifar_add

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)   

def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 10)

def get_current_classification_weight(epoch):
    return sigmoid_rampup2(epoch, 10)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch , args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr
    if args.lr_rampdown_epochs:
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_train_idxs(idxs, idxs_labeled, args, with_unlabel_if):

    idxs_unlabeled = idxs - idxs_labeled
    if args.data_argument == 'True' and args.iid != 'noniid_ssl':
        if args.label_rate == 0.1:
            idxs_train = list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.2:
            idxs_train = list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.15:
            idxs_train = list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.3:
            idxs_train = list(idxs_labeled)
        else: 
            print("error")

    elif args.data_argument == 'True' and args.iid == 'noniid_ssl' and len(list(idxs_labeled))/len(list(idxs)) < 0.4:
        idxs_train = list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)+ list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)
    else:
        idxs_train = list(idxs_labeled)
            
    if with_unlabel_if == 'with_unlabel':
        idxs_train = idxs_train + list(idxs_unlabeled)
    idxs_train = np.random.permutation(idxs_train)

    return idxs_train

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, dataset_ema = None, pseudo_label = None):
        self.dataset = dataset
        self.dataset_ema = dataset_ema
        self.idxs = list(idxs)
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]

        if self.pseudo_label != None:
            label = int(self.pseudo_label[self.idxs[item]]) 

        if self.dataset_ema != None:
            image_ema = self.dataset_ema[self.idxs[item]][0]
            return (image, image_ema), label
        else: 
            return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, dataset_ema=None, pseudo_label=None, idxs=set(), idxs_labeled=set()):
        self.args = args
        self.selected_clients = []
        self.pseudo_label = pseudo_label
        
        idxs_train = get_train_idxs(idxs, idxs_labeled, args, 'with_unlabel')
        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset, dataset_ema = dataset_ema, idxs = idxs_train, pseudo_label = pseudo_label),
            batch_size = args.local_bs
            )

    def train(self, net, args, iter_glob, user_epoch, net_ema = None, diff_w_old = None):
        net.train()
        if net_ema != None:
            net_ema.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        epoch_loss = []
        epoch_loss_ema = []
        w_t=[]
        
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 

        if args.dataset == 'cifar' and args.iid != 'noniid_tradition':
            consistency_criterion = softmax_kl_loss
        else:
            consistency_criterion = softmax_mse_loss

        residual_logit_criterion = symmetric_mse_loss


        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss_ema = []
            
            
            for batch_idx, (img, label) in enumerate(self.ldr_train):

                img, img_ema, label = img[0].to(self.args.device), img[1].to(self.args.device), label.to(self.args.device)
                adjust_learning_rate(optimizer, user_epoch * args.local_ep + iter + 1 , batch_idx, len(self.ldr_train), args)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema, volatile=True)
                target_var = torch.autograd.Variable(label)
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                if net_ema != None:
                    ema_model_out = net_ema(ema_input_var)
                else:
                    ema_model_out = net(ema_input_var)
                model_out = net(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out           
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1
                classification_weight = 1 
                class_loss = classification_weight * class_criterion(class_logit, target_var) / minibatch_size
                ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
                consistency_weight = get_current_consistency_weight(user_epoch * args.local_ep + iter + 1 )
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                loss = class_loss + consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if net_ema != None:
                    update_ema_variables(net, net_ema, args.ema_decay, user_epoch * args.local_ep + iter + 1 )
                batch_loss.append(class_loss.item())
                batch_loss_ema.append(consistency_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss_ema.append(sum(batch_loss_ema)/len(batch_loss_ema))
        if net_ema != None:
            return net.state_dict(), net_ema.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema)

    def trainc(self, net, args, iter_glob, user_epoch, net_ema = None, diff_w_old = None):
        net.train()
        if net_ema != None:
            net_ema.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        epoch_loss = []
        epoch_loss_ema = []
        w_t=[]
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
        if args.dataset == 'cifar' and args.iid != 'noniid_tradition':
            consistency_criterion = softmax_kl_loss
        else:
            consistency_criterion = softmax_mse_loss
        residual_logit_criterion = symmetric_mse_loss
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss_ema = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):
                img, img_ema, label = img[0].to(self.args.device), img[1].to(self.args.device), label.to(self.args.device)
                adjust_learning_rate(optimizer, user_epoch * args.local_ep + iter + 1 , batch_idx, len(self.ldr_train), args)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema, volatile=True)
                target_var = torch.autograd.Variable(label)
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                if net_ema != None:
                    ema_model_out = net_ema(ema_input_var)
                else:
                    ema_model_out = net(ema_input_var)
                model_out = net(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out           
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1
                classification_weight = 1 
                class_loss = classification_weight * class_criterion(class_logit, target_var) / minibatch_size
                ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
                consistency_weight = get_current_consistency_weight(user_epoch * args.local_ep + iter + 1 )
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                loss = class_loss + consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if net_ema != None:
                    if iter_glob > args.phi_g:
                        update_ema_variables(net, net_ema, args.ema_decay, user_epoch * args.local_ep + iter + 1 )
                    else:
                        update_ema_variables(net, net_ema, 0.0, user_epoch * args.local_ep + iter + 1 )
                batch_loss.append(class_loss.item())
                batch_loss_ema.append(consistency_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss_ema.append(sum(batch_loss_ema)/len(batch_loss_ema))
        if self.args.test == 2:
             return net.state_dict(), net_ema.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema), epoch_loss, epoch_loss_ema
        if int(diff_w_old) != None:
            w, w_ema = net.state_dict(), net_ema.state_dict()
            w_dic, w_ema_dic, diff_w_ema = {}, {}, {}
            comu_w, comu_w_ema= 0, 0
            w_keys = list(w.keys())    
            for i in w_keys:
                diff_w_ema[i] = ( (w[i]-w_ema[i]).float().norm(2)**2  , w[i].float().norm(2)**2  )
            if len(diff_w_ema)==33:
                diff_w_ema = cifar_add(diff_w_ema)
            else:
                diff_w_ema = mnist_add(diff_w_ema)
            for i in w_keys:
                if(iter_glob < args.phi_g):
                    w_ema_dic[i] = w_ema[i]
                    comu_w_ema += torch.numel(w_ema_dic[i])
                else:
                    if diff_w_ema[i] >= args.threshold * diff_w_old:
                        w_dic[i] = w[i]
                        comu_w += torch.numel(w_dic[i])
                    else:
                        w_ema_dic[i] = w_ema[i]
                        comu_w_ema += torch.numel(w_ema_dic[i])
            return w_dic, w_ema_dic, w_ema, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema), diff_w_ema, comu_w, comu_w_ema
        if net_ema != None:
            return net.state_dict(), net_ema.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_ema) / len(epoch_loss_ema)

class LocalUpdate_fedavg(object):
    def __init__(self, args, dataset=None, pseudo_label=None, idxs=set(), idxs_labeled=set()):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.pseudo_label=pseudo_label

        if idxs_labeled == set():
            idxs_train = np.random.permutation(list(idxs))
        else:
            idxs_train = get_train_idxs(idxs, idxs_labeled, args, 'without_unlabel')

        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset, idxs = idxs_train, pseudo_label = pseudo_label),
            batch_size=self.args.local_bs, 
            shuffle=True
            )

    def train(self, net):

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):   
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)




class LocalUpdate_fedfixmatch(object):
    def __init__(self, args, dataset_strong=None, dataset_weak=None, pseudo_label=None, idxs=set(), idxs_labeled=set()):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.pseudo_label=pseudo_label
        idxs_train = get_train_idxs(idxs, idxs_labeled, args, 'with_unlabel')
        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset_weak, dataset_ema = dataset_strong, idxs = idxs_train, pseudo_label = pseudo_label),
            batch_size=self.args.local_bs
            )

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=False)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):   
                img, img_ema, label = img[0].to(self.args.device), img[1].to(self.args.device), label.to(self.args.device)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema)
                target_var = torch.autograd.Variable(label)                
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                ema_model_out = net(ema_input_var)
                model_out = net(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out           
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1
                class_loss = class_criterion(class_logit, target_var) / minibatch_size
                ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
                pseudo_label1 = torch.softmax(model_out.detach_(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label1, dim=-1)
                mask = max_probs.ge(self.args.threshold_pl).float()
                Lu = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask).mean()
                loss = class_loss + Lu 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)





class LocalUpdate_fedmatch(object):#local
    def __init__(self, args, dataset_strong=None, dataset_weak=None, pseudo_label=None, idxs=set(), idxs_labeled=set()):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.pseudo_label=pseudo_label

        idxs_train = get_train_idxs(idxs, idxs_labeled, args, 'with_unlabel')
        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset_weak, dataset_ema = dataset_strong, idxs = idxs_train, pseudo_label = pseudo_label),
            batch_size=self.args.local_bs
            )

    def train(self, net, net_helper_1, net_helper_2):

        net.train()
        net_helper_1.train()
        net_helper_2.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=False)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
        epoch_loss = []

        for iter in range(self.args.local_ep):

            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):   
                img, img_ema, label = img[0].to(self.args.device), img[1].to(self.args.device), label.to(self.args.device)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema)
                target_var = torch.autograd.Variable(label)
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                ema_model_out = net(ema_input_var)
                model_out = net(input_var)
                model_out_helper_1 = net_helper_1(input_var)
                model_out_helper_2 = net_helper_2(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out     
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1
                class_loss = class_criterion(class_logit, target_var) / minibatch_size
                pseudo_label1 = torch.softmax(model_out.detach_(), dim=-1)
                pseudo_label2 = torch.softmax(model_out_helper_1.detach_(), dim=-1)
                pseudo_label3 = torch.softmax(model_out_helper_2.detach_(), dim=-1)
                max_probs1, targets_u1 = torch.max(pseudo_label1, dim=-1)
                max_probs2, targets_u2 = torch.max(pseudo_label2, dim=-1)
                max_probs3, targets_u3 = torch.max(pseudo_label3, dim=-1)

                if torch.equal(targets_u1, targets_u2) and torch.equal(targets_u1, targets_u3):
                    max_probs = torch.max(max_probs1, max_probs2)
                    max_probs = torch.max(max_probs, max_probs3)
                else: 
                    max_probs = max_probs1 - 0.2
                targets_u = targets_u1
                mask = max_probs.ge(self.args.threshold_pl).float()
                Lu = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask).mean()
                loss = class_loss + Lu 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



