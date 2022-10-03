

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd
from src.ProteinsDataset import *
import scipy.optimize
from src.utils import *



def accuracy(batch, output, onehot=False):
    bs = output.shape[1]
    ra = range(bs)
    if onehot==False:
        proteinOUT1 = batch[1][1:-1,:]
        proteinOUT1 = proteinOUT1.float().t()
        
        proteinOUT2 = output.max(dim=2)[1][:-1,:]
        proteinOUT2 = proteinOUT2.float().t()

    Distance = torch.cdist(proteinOUT1, proteinOUT2, p=0.0)[ra,ra]
    return torch.sum(Distance)



def ConditionalSquaredEntropyMatchingLoss(batch,
                                  model,
                                  CCL_mean,
                                  device,
                                  samplingMultiple=1,
                                  gumbel=True):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    output = model(inp_data, target[:-1, :])
    acc = accuracy(batch, output, onehot=model.onehot).item()
    output = output.reshape(-1, output.shape[2])
    if model.onehot:
        _, targets_Original = target.max(dim=2)
    else:
        targets_Original= target
    
    targets_Original = targets_Original[1:].reshape(-1)
    lossCE = CCL_mean(output, targets_Original)

    ### fake step
    if gumbel==False:
        samples = model.pseudosample(inp_data, target, nsample=1, method="simple")
        samples = samples.max(dim=2)[1]


    if gumbel ==True:
        samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
        _, samples_Original = samples.max(dim=2)
    else: 
        samples_Original = samples
    output = model(inp_data, samples[:-1, :])
    output = output.reshape(-1, output.shape[2])
    samples_Original = samples_Original[1:].reshape(-1)
    lossEntropy = torch.exp(-1*CCL_mean(output, samples_Original))
    for i in range(samplingMultiple-1):
        samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
        if gumbel==False:
            samples = samples.max(dim=2)[1]
        output = model(inp_data, samples[:-1, :])
        output = output.reshape(-1, output.shape[2])
        if gumbel ==True:
            _, samples_Original = samples.max(dim=2)
        else: 
            samples_Original = samples
        samples_Original = samples_Original[1:].reshape(-1)
        lossEntropy += torch.exp(-1*CCL_mean(output, samples_Original))
    
    lossEntropy = lossEntropy/samplingMultiple  

    
    return lossCE, lossEntropy, acc






def ReyniMatchingLossNew(batch,
                                    model,
                                    criterion,
                                    criterionMatching,
                                    device,
                                    accumulate=False,
                                    ncontrastive=5,
                                    sampler="gumbel"):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]

        
    lossMatrix = torch.zeros((bs,ncontrastive+1)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([0]*bs).to(device)
    
    
    
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])
    targets_Original = target
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,0] = loss
    LossCE += loss.mean()
    
    for i in range(1,ncontrastive+1):
        contrastiveTarget = model.pseudosample(inp_data, target, nsample=1, method=sampler)
        output2 = model(inp_data, contrastiveTarget[:-1, :])
        output2 = output2.reshape(-1, output2.shape[2])
        _, targets_Original2 = contrastiveTarget.max(dim=2)
        targets_Original2 = targets_Original2[1:].reshape(-1)
        loss2 = criterion(output2, targets_Original2).reshape(-1,bs).mean(dim=0)
        lossMatrix[:,i] = loss2
        
    lossMatrix *=-1
    lossMatching = -1*criterionMatching(lossMatrix, targetMatching)

    return LossCE, lossMatching


def LLLoss(batch,
           model,
           criterion, onehot):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])#keep last dimension
        if onehot:
            _, targets_Original = target.max(dim=2)
        else:
            targets_Original= target
        targets_Original = targets_Original[1:].reshape(-1)
        loss = criterion(output, targets_Original)
        return loss




















        