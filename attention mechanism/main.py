#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from narm import NARM
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/recsys15/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size') ###recsys19 512   15 512
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')  ##recsys151_4 30 epoch   19 50
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', default=1,action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)
    
    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    n_items = 37484

    if args.test:
        for i in range(5):
            results=np.zeros((3,3))
            model = NARM(n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device)
            optimizer = optim.Adam(model.parameters(), args.lr)
            criterion = nn.CrossEntropyLoss()
            scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
            for epoch in tqdm(range(args.epoch)):
                # train for one epoch
                scheduler.step(epoch=epoch)
                trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=1000)
            model.eval()
            recalls5,recalls10,recalls20 = [],[],[]
            mrrs5,mrrs10,mrrs20 = [],[],[]
            ndcgs5,ndcgs10,ndcgs20 = [],[],[]
            with torch.no_grad():
                for seq, target, lens in tqdm(valid_loader):
                    seq = seq.to(device)
                    target = target.to(device)
                    outputs = model(seq, lens)
                    logits = F.softmax(outputs, dim=1)
                    recall5, mrr5, ndcg5 = metric.evaluate(logits, target, k=5)
                    recall10, mrr10, ndcg10 = metric.evaluate(logits, target, k=10)
                    recall20, mrr20, ndcg20 = metric.evaluate(logits, target, k=args.topk)
                    recalls5.append(recall5)
                    mrrs5.append(mrr5)
                    ndcgs5.append(ndcg5)
                    recalls10.append(recall10)
                    mrrs10.append(mrr10)
                    ndcgs10.append(ndcg10)
                    recalls20.append(recall20)
                    mrrs20.append(mrr20)
                    ndcgs20.append(ndcg20)


            results[0,0]=np.mean(recalls5)
            results[0,1]=np.mean(mrrs5)
            results[0,2]=np.mean(ndcgs5)
            results[1, 0] = np.mean(recalls10)
            results[1, 1] = np.mean(mrrs10)
            results[1, 2] = np.mean(ndcgs10)
            results[2, 0] = np.mean(recalls20)
            results[2, 1] = np.mean(mrrs20)
            results[2, 2] = np.mean(ndcgs20)

            with open('recsys19/test_performances_on.txt', 'a') as f:
                f.write( str(results) + '\n')

    model = NARM(n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 1000)

        recall, mrr ,ndcg= validate(test_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk, mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    ndcgs=[]
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr,ndcg = metric.evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_ndcg=np.mean(ndcgs)
    return mean_recall, mean_mrr, mean_ndcg


if __name__ == '__main__':
    main()
