#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=1, help="for test")
    parser.add_argument('--label_rate', type=float, default=0.1, help="the fraction of labeled data")
    parser.add_argument('--data_argument', type=str, default='True', help="data argumentation")
    parser.add_argument('--comu_rate',type=float, default=0.5,help="the comu_rate of ema model")
    parser.add_argument('--ramp',type=str,default='linear', help="ramp of comu")
    parser.add_argument('--threshold', type=float, default=1.0, help="threshold of cutting w")  
    parser.add_argument('--threshold_pl', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--lambda-u', default=1, type=float,help='coefficient of unlabeled loss')
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--p1epochs', type=int, default=20, help="p1rounds of training")
    parser.add_argument('--p2epochs', type=int, default=10, help="p2rounds of training")
    parser.add_argument('--phi_g', type=int, default=10, help="tipping point 1")
    parser.add_argument('--psi_g', type=int, default=40, help="tipping point 2")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--initial-lr', default=0.0, type=float,metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=str, default ='noniid_ssl', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    args = parser.parse_args()
    return args
