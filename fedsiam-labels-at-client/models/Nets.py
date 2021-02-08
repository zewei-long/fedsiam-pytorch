#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self,args):
        super(CNNCifar, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def mnist_add(diff_w_ema):
    diff_w_ema1 = {}

    diff_w_ema1['conv1.weight'] = torch.sqrt(diff_w_ema['conv1.weight'][0] + diff_w_ema['conv1.bias'][0]) / torch.sqrt(diff_w_ema['conv1.weight'][1] + diff_w_ema['conv1.bias'][1])
    diff_w_ema1['conv1.bias'] = torch.sqrt(diff_w_ema['conv1.weight'][0] + diff_w_ema['conv1.bias'][0]) / torch.sqrt(diff_w_ema['conv1.weight'][1] + diff_w_ema['conv1.bias'][1])

    diff_w_ema1['conv2.weight'] = torch.sqrt(diff_w_ema['conv2.weight'][0] + diff_w_ema['conv2.bias'][0]) / torch.sqrt(diff_w_ema['conv2.weight'][1] + diff_w_ema['conv2.bias'][1])
    diff_w_ema1['conv2.bias'] = torch.sqrt(diff_w_ema['conv2.weight'][0] + diff_w_ema['conv2.bias'][0]) / torch.sqrt(diff_w_ema['conv2.weight'][1] + diff_w_ema['conv2.bias'][1]) 

    diff_w_ema1['fc1.weight'] = torch.sqrt(diff_w_ema['fc1.weight'][0] + diff_w_ema['fc1.bias'][0]) / torch.sqrt(diff_w_ema['fc1.weight'][1] + diff_w_ema['fc1.bias'][1])
    diff_w_ema1['fc1.bias'] = torch.sqrt(diff_w_ema['fc1.weight'][0] + diff_w_ema['fc1.bias'][0]) / torch.sqrt(diff_w_ema['fc1.weight'][1] + diff_w_ema['fc1.bias'][1])

    diff_w_ema1['fc2.weight'] = torch.sqrt(diff_w_ema['fc2.weight'][0] + diff_w_ema['fc2.bias'][0]) / torch.sqrt(diff_w_ema['fc2.weight'][1] + diff_w_ema['fc2.bias'][1])
    diff_w_ema1['fc2.bias'] = torch.sqrt(diff_w_ema['fc2.weight'][0] + diff_w_ema['fc2.bias'][0]) / torch.sqrt(diff_w_ema['fc2.weight'][1] + diff_w_ema['fc2.bias'][1]) 

    return diff_w_ema1

def cifar_add(diff_w_ema):
    diff_w_ema1 = {}

    diff_w_ema1['conv_layer.0.weight'] = torch.sqrt(diff_w_ema['conv_layer.0.weight'][0] + diff_w_ema['conv_layer.0.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.0.weight'][1] + diff_w_ema['conv_layer.0.bias'][1])
    diff_w_ema1['conv_layer.0.bias'] = diff_w_ema1['conv_layer.0.weight']

    diff_w_ema1['conv_layer.1.weight'] = torch.sqrt(diff_w_ema['conv_layer.1.weight'][0] + diff_w_ema['conv_layer.1.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.1.weight'][1] + diff_w_ema['conv_layer.1.bias'][1])
    diff_w_ema1['conv_layer.1.bias'] = diff_w_ema1['conv_layer.1.weight']
    diff_w_ema1['conv_layer.1.running_mean'] = diff_w_ema1['conv_layer.1.weight']
    diff_w_ema1['conv_layer.1.running_var'] = diff_w_ema1['conv_layer.1.weight']
    diff_w_ema1['conv_layer.1.num_batches_tracked'] = diff_w_ema1['conv_layer.1.weight']

    diff_w_ema1['conv_layer.3.weight'] = torch.sqrt(diff_w_ema['conv_layer.3.weight'][0] + diff_w_ema['conv_layer.3.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.3.weight'][1] + diff_w_ema['conv_layer.3.bias'][1])
    diff_w_ema1['conv_layer.3.bias'] = diff_w_ema1['conv_layer.3.weight']

    diff_w_ema1['conv_layer.6.weight'] = torch.sqrt(diff_w_ema['conv_layer.6.weight'][0] + diff_w_ema['conv_layer.6.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.6.weight'][1] + diff_w_ema['conv_layer.6.bias'][1])
    diff_w_ema1['conv_layer.6.bias'] = diff_w_ema1['conv_layer.6.weight']

    diff_w_ema1['conv_layer.7.weight'] = torch.sqrt(diff_w_ema['conv_layer.7.weight'][0] + diff_w_ema['conv_layer.7.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.7.weight'][1] + diff_w_ema['conv_layer.7.bias'][1])
    diff_w_ema1['conv_layer.7.bias'] = diff_w_ema1['conv_layer.7.weight']
    diff_w_ema1['conv_layer.7.running_mean'] = diff_w_ema1['conv_layer.7.weight']
    diff_w_ema1['conv_layer.7.running_var'] = diff_w_ema1['conv_layer.7.weight']
    diff_w_ema1['conv_layer.7.num_batches_tracked'] = diff_w_ema1['conv_layer.7.weight']

    diff_w_ema1['conv_layer.9.weight'] = torch.sqrt(diff_w_ema['conv_layer.9.weight'][0] + diff_w_ema['conv_layer.9.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.9.weight'][1] + diff_w_ema['conv_layer.9.bias'][1])
    diff_w_ema1['conv_layer.9.bias'] = diff_w_ema1['conv_layer.9.weight']


    diff_w_ema1['conv_layer.13.weight'] = torch.sqrt(diff_w_ema['conv_layer.13.weight'][0] + diff_w_ema['conv_layer.13.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.13.weight'][1] + diff_w_ema['conv_layer.13.bias'][1])
    diff_w_ema1['conv_layer.13.bias'] = diff_w_ema1['conv_layer.13.weight']

    diff_w_ema1['conv_layer.14.weight'] = torch.sqrt(diff_w_ema['conv_layer.14.weight'][0] + diff_w_ema['conv_layer.14.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.14.weight'][1] + diff_w_ema['conv_layer.14.bias'][1])
    diff_w_ema1['conv_layer.14.bias'] = diff_w_ema1['conv_layer.14.weight']
    diff_w_ema1['conv_layer.14.running_mean'] = diff_w_ema1['conv_layer.14.weight']
    diff_w_ema1['conv_layer.14.running_var'] = diff_w_ema1['conv_layer.14.weight']
    diff_w_ema1['conv_layer.14.num_batches_tracked'] = diff_w_ema1['conv_layer.14.weight']

    diff_w_ema1['conv_layer.16.weight'] = torch.sqrt(diff_w_ema['conv_layer.16.weight'][0] + diff_w_ema['conv_layer.16.bias'][0]) / torch.sqrt(diff_w_ema['conv_layer.16.weight'][1] + diff_w_ema['conv_layer.16.bias'][1])
    diff_w_ema1['conv_layer.16.bias'] = diff_w_ema1['conv_layer.16.weight']

    diff_w_ema1['fc_layer.1.weight'] = torch.sqrt(diff_w_ema['fc_layer.1.weight'][0] + diff_w_ema['fc_layer.1.bias'][0]) / torch.sqrt(diff_w_ema['fc_layer.1.weight'][1] + diff_w_ema['fc_layer.1.bias'][1])
    diff_w_ema1['fc_layer.1.bias'] = diff_w_ema1['fc_layer.1.weight']

    diff_w_ema1['fc_layer.3.weight'] = torch.sqrt(diff_w_ema['fc_layer.3.weight'][0] + diff_w_ema['fc_layer.3.bias'][0]) / torch.sqrt(diff_w_ema['fc_layer.3.weight'][1] + diff_w_ema['fc_layer.3.bias'][1])
    diff_w_ema1['fc_layer.3.bias'] = diff_w_ema1['fc_layer.3.weight']

    diff_w_ema1['fc_layer.6.weight'] = torch.sqrt(diff_w_ema['fc_layer.6.weight'][0] + diff_w_ema['fc_layer.6.bias'][0]) / torch.sqrt(diff_w_ema['fc_layer.6.weight'][1] + diff_w_ema['fc_layer.6.bias'][1])
    diff_w_ema1['fc_layer.6.bias'] = diff_w_ema1['fc_layer.6.weight']

    return diff_w_ema1