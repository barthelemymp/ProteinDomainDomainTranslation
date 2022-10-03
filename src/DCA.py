import subprocess
import matplotlib.pyplot as plt


import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import wandb
sys.path.append("")
from src.ProteinTransformer import *
from src.ProteinsDataset import *
from src.MatchingLoss import *
from src.utils import *
from src.ardca import *




def PPV_from_pds(pds, pdblist, chain1list, chain2list, hmmRadical, mode ="inter"):

    tempFile_raw=next(tempfile._get_candidate_names())
    tempFile= tempFile_raw+".npy"
    mode = "inter"
    #### getlist
    pdbpath = tempFile_raw+"pdblisttemp.npy"
    chain1path = tempFile_raw+"chain1listtemp.npy"
    chain2path = tempFile_raw+"chain2listtemp.npy"
    np.save(pdbpath, pdblist)
    np.save(chain1path, chain1list)
    np.save(chain2path, chain2list)
    tempTrainr = writefastafrompds(pds)
    tempTrain=tempTrainr+"joined.faa"
    output = subprocess.check_output(["stdbuf", "-oL", "julia", "contactPlot_merged.jl", tempTrain,pdbpath, chain1path, chain2path, hmmRadical, tempFile, mode])
    print(output)
    ppv = np.load(tempFile)
    os.remove(pdbpath)
    os.remove(chain1path)
    os.remove(chain2path)
    os.remove(tempTrain)
    os.remove(tempFile)
    return ppv

def sampleDataset(model, pds, len_output, multiplicative =1):
    max_len = len_output
    pds_sample = copy.deepcopy(pds)

    batchIndex = makebatchList(len(pds_sample), 100)
    for batchI in batchIndex:
        sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
        pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
    for mu in range(multiplicative-1):
        for batchI in batchIndex:
            sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
            pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
            pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
    return pds_sample

        
        

    
    
    
    
    




















