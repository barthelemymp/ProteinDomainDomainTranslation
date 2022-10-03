import torch
from torchtext.data.metrics import bleu_score
import sys
import pandas as pd
import numpy as np
import math
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ProteinsDataset import *
from tqdm import tqdm


def buildhmm(hmmout, ali):
    subprocess.run(["hmmbuild", "--symfrac","0.0", hmmout, ali])




def getlists(df, fam):
    pdblist = list(df[df["id"]==fam]["pdb"])
    chain1list = list(df[df["id"]==fam]["chain1"])
    chain2list = list(df[df["id"]==fam]["chain2"])
    return pdblist, chain1list, chain2list



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def splitcsv(pathToCSV, name, repartition, shuffle=False, maxval=None):
    
    train_per = repartition[0]
    test_per = repartition[1]
    if len(repartition)>=3:
        val_per = repartition[2]
        
    path = pathToCSV + name + ".csv"
    
    df = pd.read_csv(path)
    if shuffle == True:
        df = df.sample(frac=1).reset_index(drop=True)
    
    total_size=len(df)
    
    train_size=math.floor(train_per*total_size)
    test_size=math.floor(test_per*total_size)
    if maxval:
        test_size= min(test_size, maxval)
    traintest=df.head(train_size+test_size)
    train = traintest.head(train_size)
    test= traintest.tail(test_size)
    
    train.to_csv(pathToCSV + name +'_train.csv', index = False)
    test.to_csv(pathToCSV + name +'_test.csv', index = False)
    
    if len(repartition)>=3:
        val_size=math.floor(val_per*total_size)
        if maxval:
            val_size= min(val_size, maxval)
        val=df.tail(val_size)
        val.to_csv(pathToCSV + name +'_val.csv',index = False)
    

def getLengthfromCSV(pathToFile):
    df = pd.read_csv(pathToFile)
    inputsize = len(df.iloc[1][0].split(" "))
    outputsize = len(df.iloc[1][1].split(" "))
    return inputsize, outputsize
    
def HungarianMatching(pds_val, model):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.padIndex, reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    bs = listin.shape[1]
    scoreHungarian = np.zeros((bs, bs))
    with torch.no_grad():
        for j in tqdm(range(bs)):

            inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,bs,1)
            output = model(inp_repeted, listout[:-1, :])
            output = output.reshape(-1, output.shape[2])#keep last dimension
            _, targets_Original = listout.max(dim=2)
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterionE(output, targets_Original).reshape(-1,bs).mean(dim=0)
            scoreHungarian[j,:] = loss.cpu().numpy()
    return scoreHungarian

def makebatchList(tot, bs):
    batchIndex=[]
    nbatch = tot//bs
    last = tot%bs
    for i in range(nbatch):
        start = i*bs
        end =start+bs
        batchIndex.append(range(start, end))
    if last!=0:
        start = nbatch*bs
        end = start+last
        batchIndex.append(range(start, end))
    return batchIndex
        
    

def HungarianMatchingBS(pds_val, model, batchs):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.padIndex, reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    tot = listin.shape[1]
    scoreHungarian = np.zeros((tot, tot))
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        for j in tqdm(range(tot)):
            for batch in batchIndex:
                if model.onehot:
                    _, targets_Original = listout[:,batch].max(dim=2)
                    inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,len(batch),1)
                else:
                    targets_Original= listout[:,batch]
                    inp_repeted = listin[:,j].unsqueeze(1).repeat(1,len(batch))
                
                output = model(inp_repeted, listout[:-1, batch])
                output = output.reshape(-1, output.shape[2])#keep last dimension

                targets_Original = targets_Original[1:].reshape(-1)
                loss = criterionE(output, targets_Original).reshape(-1,len(batch)).mean(dim=0)
                scoreHungarian[j,batch] = loss.cpu().numpy()

    return scoreHungarian







def distanceTrainVal(pds_val, pds_train):
    if pds_val.onehot:
        proteinIN1 = pds_train[:][0].max(dim=2)[1].float().t()
        proteinOUT1 = pds_train[:][1].max(dim=2)[1].float().t()
        proteinIN2 = pds_val[:][0].max(dim=2)[1].float().t()
        proteinOUT2 = pds_val[:][1].max(dim=2)[1].float().t()
    else:
        proteinIN1 = pds_train[:][0].float().t()
        proteinOUT1 = pds_train[:][1].float().t()
        proteinIN2 = pds_val[:][0].float().t()
        proteinOUT2 = pds_val[:][1].float().t()
    DistanceIN = torch.cdist(proteinIN2, proteinIN1, p=0.0)
    DistanceOUT = torch.cdist(proteinOUT2, proteinOUT1, p=0.0)
    return DistanceIN, DistanceOUT
    
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
    
def accuracyMatrix(batch, output, onehot=False):
    bs = output.shape[1]
    ra = range(bs)
    if onehot==False:
        proteinOUT1 = batch[1][1:-1,:]
        proteinOUT1 = proteinOUT1.float().t()
        
        proteinOUT2 = output.max(dim=2)[1][:-1,:]
        proteinOUT2 = proteinOUT2.float().t()

    Distance = torch.cdist(proteinOUT1, proteinOUT2, p=0.0)[ra,ra]
    return Distance

        
def ConditionalEntropyEstimator(pds_val, model, batchs=100, returnAcc=False):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.padIndex, reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    tot = listin.shape[1]
    max_len = listout.shape[0]
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        entropylist =[]
        acc = 0
        for batch in tqdm(batchIndex):
            if model.onehot:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                output = model(listin[:,batch], sampled[:-1, :])
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]
                targets_Original = targets_Original[1:].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).mean()
                entropylist.append(Entropy)
            else:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                acc+=accuracy(pds_val[batch], sampled[1:]).item()
                output = model(listin[:,batch], sampled[:-1, :])[:-1]
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]#listout[:,batch]
                targets_Original = targets_Original[1:-1].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).sum(dim=0)
                entropylist+=[Entropy[i] for i in range(len(Entropy))]
        print(acc/len(pds_val))
        meanEntropy = sum(entropylist)/len(entropylist)
    if returnAcc:
        return meanEntropy, acc/len(pds_val)
    else:
        return meanEntropy
    
    
   
def ConditionalEntropyEstimatorGivenInp(inp, model, pad, max_len,nseq=1000, batchs=1000, returnAcc=False):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pad, reduction='none')
    listin = inp.unsqueeze(1).repeat(1,nseq)
    tot = listin.shape[1]
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        entropylist =[]
        acc = 0
        for batch in tqdm(batchIndex):
            if model.onehot:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                output = model(listin[:,batch], sampled[:-1, :])
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]
                targets_Original = targets_Original[1:].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).mean()
                entropylist.append(Entropy)
            else:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                output = model(listin[:,batch], sampled[:-1, :])[:-1]
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]#listout[:,batch]
                targets_Original = targets_Original[1:-1].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).sum(dim=0)
                entropylist += [Entropy[i] for i in range(len(Entropy))]
        meanEntropy = sum(entropylist)/len(entropylist)
    if returnAcc:
        return meanEntropy, acc/nseq
    else:
        return meanEntropy
    

        
def sample_dataset(pds, model, nrepet=1, pathtosave=None): 
    pds_sample = copy.deepcopy(pds)
    max_len=pds.tensorOUT.shape[0]
    batchIndex = makebatchList(len(pds_sample), 300)
    for batchI in batchIndex:
        sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
        pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
    for i in range(1, nrepet):
        for batchI in batchIndex:
            sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
            pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
            pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
    if pathtosave:
        path = pathtosave+"joined.faa"
        writefasta(torch.cat([torch.nn.functional.one_hot(pds_sample.tensorIN, num_classes=model.trg_vocab_size), torch.nn.functional.one_hot(pds_sample.tensorOUT, num_classes=model.trg_vocab_size)]), path, mapstring =pds_sample.mapstring)
        path = pathtosave+"_2.faa"
        writefasta(torch.nn.functional.one_hot(pds_sample.tensorOUT, num_classes=model.trg_vocab_size), path, mapstring =pds_sample.mapstring)
