
import tempfile
import os
import subprocess
import torch
import numpy as np
import scipy.optimize


def back2seq(seq, mapstring, unk="-", onehot=True):
    # print(onehot)
    BackSymbolMap=dict([(i,mapstring[i]) for i in range(len(mapstring))])
    BackSymbolMap[len(mapstring)] =unk
    BackSymbolMap[len(mapstring)+1] ="<sos>"
    BackSymbolMap[len(mapstring)+2] = "<eos>"
    BackSymbolMap[len(mapstring)+3] = "<pad>"
    if onehot:    
        seq_Original = seq.max(dim=1)[1].cpu().numpy()
    else:
        seq_Original = seq.cpu().numpy()
    seqout = ""
    for i in range(seq_Original.shape[0]):
        if seq_Original[i]<=len(mapstring):
            seqout+=BackSymbolMap[seq_Original[i]]
        else:
            seqout+=unk
    return seqout
    

def writefasta(matrix, destination, mapstring = "-ACDEFGHIKLMNPQRSTVWY", onehot=True):
    ofile = open(destination, "w")
    for i in range(matrix.shape[1]):
        seq = back2seq(matrix[:,i], mapstring, onehot=onehot)
        name = str(i)
        ofile.write(">" + name + "\n" +seq + "\n")
    ofile.close()
    

def writefastafrompds(pds):
    tempFile=next(tempfile._get_candidate_names())
    tempfileIN = tempFile+"1.faa"
    tempfileOUT = tempFile+"2.faa"
    tempfilejoined = tempFile+"joined.faa"
    writefasta(pds.tensorIN, tempfileIN, mapstring =pds.mapstring, onehot = pds.onehot)
    writefasta(pds.tensorOUT, tempfileOUT, mapstring =pds.mapstring, onehot = pds.onehot)
    writefasta(torch.cat([pds.tensorIN, pds.tensorOUT]), tempfilejoined, mapstring =pds.mapstring, onehot = pds.onehot)
    return tempFile

def removetemp(temp):
    os.remove(temp+"1.faa")
    os.remove(temp+"2.faa")
    os.remove(temp+"joined.faa")

def ARDCA(pdsTrain, pdsTest, pdsVal):
    tempTrain = writefastafrompds(pdsTrain)
    tempTest = writefastafrompds(pdsTest)
    tempVal = writefastafrompds(pdsVal)
    tempScoreH = next(tempfile._get_candidate_names())
    os.system("export JULIA_NUM_THREADS=$(nproc --all)")
    output = subprocess.check_output(["julia", "ardca_call.jl", tempTrain, tempTest, tempVal, tempScoreH])
    print(output)
    removetemp(tempTrain)
    removetemp(tempTest)
    removetemp(tempVal)
    tt = str(output).split('\\n')[-3].split('=')[-1].split(',')
    CEtrain = float(tt[0].split('(')[-1])
    CEtest = float(tt[1])
    CEval = float(tt[2].split(')')[0])
    ttacc = str(output).split('\\n')[-2].split('=')[-1].split(',')
    acctrain = float(ttacc[0].split('(')[-1])
    acctest = float(ttacc[1])
    accval = float(ttacc[2].split(')')[0])
    scoreHungarianVal = np.load(tempScoreH)
    scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
    scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
    return CEtrain, CEtest, CEval, acctrain, acctest, accval, scoreMatchingVal
    
    
def ARDCA_returnmatrix(pdsTrain, pdsTest, pdsVal):
    tempTrain = writefastafrompds(pdsTrain)
    tempTest = writefastafrompds(pdsTest)
    tempVal = writefastafrompds(pdsVal)
    tempScoreH = next(tempfile._get_candidate_names())
    os.system("export JULIA_NUM_THREADS=$(nproc --all)")
    output = subprocess.check_output(["julia", "ardca_call.jl", tempTrain, tempTest, tempVal, tempScoreH])
    print(output)
    removetemp(tempTrain)
    removetemp(tempTest)
    removetemp(tempVal)
    tt = str(output).split('\\n')[-3].split('=')[-1].split(',')
    CEtrain = float(tt[0].split('(')[-1])
    CEtest = float(tt[1])
    CEval = float(tt[2].split(')')[0])
    ttacc = str(output).split('\\n')[-2].split('=')[-1].split(',')
    acctrain = float(ttacc[0].split('(')[-1])
    acctest = float(ttacc[1])
    accval = float(ttacc[2].split(')')[0])
    scoreHungarianVal = np.load(tempScoreH)
    scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
    scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
    os.remove(tempScoreH)
    return scoreHungarianVal
    
def ARDCA_saveAllmatrix(pdsTrain, pdsVal, tempScoreH, tempScoreAcc, tempScoreCE):
    tempTrain = writefastafrompds(pdsTrain)
    tempVal = writefastafrompds(pdsVal)
    os.system("export JULIA_NUM_THREADS=$(nproc --all)")
    output = subprocess.check_output(["julia", "ardca_returnM.jl", tempTrain, tempVal, tempScoreH, tempScoreAcc, tempScoreCE])
    print(output)
    removetemp(tempTrain)
    removetemp(tempVal)
    return 
    
def ARDCA_timeit(pdsTrain, pdsVal):
    tempTrain = writefastafrompds(pdsTrain)
    tempVal = writefastafrompds(pdsVal)
    tempScoreH = next(tempfile._get_candidate_names())
    os.system("export JULIA_NUM_THREADS=$(nproc --all)")
    output = subprocess.check_output(["julia", "ardca_train.jl", tempTrain, tempVal])
    print(output)
    return output



def ARDCA_returnCE(pdsTrain, pdsVal):
    tempTrain = writefastafrompds(pdsTrain)
    tempVal = writefastafrompds(pdsVal)
    tempScoreH = next(tempfile._get_candidate_names())
    os.system("export JULIA_NUM_THREADS=$(nproc --all)")
    output = subprocess.check_output(["julia", "ardca_plot.jl", tempTrain, tempVal, tempScoreH])
    removetemp(tempTrain)
    removetemp(tempVal)
    scoreCE = np.load(tempScoreH)

    return scoreCE
