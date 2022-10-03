import argparse
import numpy as np
import argparse
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import wandb
from src.ProteinTransformer import *
from src.ProteinsDataset import *
from src.MatchingLoss import *
from src.utils import *
from src.ardca import *
from src.DCA import *
import json
def get_params(params):

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--trainset', type=str, help="Path to the dataset for training")
    parser.add_argument('--valset', type=str, help="Path to the dataset for validation")
    parser.add_argument('--save', type=str, default="", help="path to save model, if empty will not save")
    parser.add_argument('--load', type=str, default="", help="path to load model, if empty will not save")
    parser.add_argument('--modelconfig', type=str, default="shallow.config.json", help="hyperparameter")
    parser.add_argument('--outputfile', type=str, default="output.txt", help="file to print scores")
    
    args = parser.parse_args(params)

    return args

def main(params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Params
    opts = get_params(params)
    f = open(opts.outputfile, "w")
    if opts.save:
        save_model=True
        model_path_save = opts.save
    else:
        save_model = False
    if opts.load:
        load_model=True
        model_path_load = opts.load
    else:
        load_model = False
        
    with open(opts.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)
        
    onehot= False
    Unalign=modelconfig["Unalign"]
    pds_train = ProteinTranslationDataset(opts.trainset, device=device, Unalign=modelconfig["Unalign"], filteringOption='and', returnIndex=True, onehot=onehot)
    pds_val = ProteinTranslationDataset(opts.valset, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
    len_input = pds_train.inputsize
    len_output = pds_train.outputsize
    
    ntrain = len(pds_train)
    nval = len(pds_val)

    dval1,dval2 = distanceTrainVal(pds_train, pds_val)
    print("median", (dval1+dval2).min(dim=0)[0].median())
    maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
    maskValclose = maskValclose.cpu().numpy()
    maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
    maskValfar = maskValfar.cpu().numpy()
    
    train_iterator = DataLoader(pds_train, batch_size=modelconfig["batch_size"],
                    shuffle=True, num_workers=0, collate_fn=default_collate)
    val_iterator = DataLoader(pds_val, batch_size=modelconfig["batch_size"],
                    shuffle=True, num_workers=0, collate_fn=default_collate)
    
    src_pad_idx =pds_train.padIndex
    src_position_embedding = PositionalEncoding(modelconfig["embedding_size"], max_len=len_input,device=device)
    trg_position_embedding = PositionalEncoding(modelconfig["embedding_size"], max_len=len_output, device=device)
            
           
    model = Transformer(
        modelconfig["embedding_size"],
        modelconfig["src_vocab_size"],
        modelconfig["trg_vocab_size"],
        src_pad_idx,
        modelconfig["num_heads"],
        modelconfig["num_encoder_layers"],
        modelconfig["num_decoder_layers"],
        modelconfig["forward_expansion"],
        modelconfig["dropout"],
        src_position_embedding,
        trg_position_embedding,
        device,
        onehot=onehot,
    ).to(device)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
 
    optimizer = optim.AdamW(model.parameters(), lr=modelconfig["learning_rate"], weight_decay=modelconfig["wd"])

    if load_model:
        load_checkpoint(torch.load(model_path_load), model, optimizer)
    
    
    criterion = nn.CrossEntropyLoss(ignore_index=pds_train.padIndex)
    criterion_raw = nn.CrossEntropyLoss(reduction='none')
    criterionMatching = nn.CrossEntropyLoss()
    num_epochs = modelconfig["num_epochs"]
    for epoch in range(modelconfig["num_epochs"]+1):
        print(f"[Epoch {epoch} / {num_epochs}]")
        model.train()
        lossesCE = []
        for batch_idx, batch in enumerate(train_iterator):
            if modelconfig["use_entropic"]==False:
                optimizer.zero_grad()
                loss = LLLoss(batch, model, criterion, onehot)
                lossesCE.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                optimizer.step()
                
            else:
                optimizer.zero_grad()
                lossCE, lossEntropy = ReyniMatchingLossNew(batch, 
                                                    model,
                                                    criterion_raw,
                                                    criterionMatching,
                                                    device,
                                                    accumulate=False,
                                                    ncontrastive=modelconfig["ncontrastive"],
                                                    sampler="gumbel")
                lossesCE.append(lossCE.item())
                loss = lossCE + modelconfig["alpha_entropic"] * lossEntropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                optimizer.step()
            
        mean_lossCETrain = sum(lossesCE) / len(lossesCE)

        model.eval()
        lossesCE_eval = []
        lossesMatching_eval = []
        accuracyVal = 0
        

        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                inp_data, target= batch[0], batch[1]
                inp_data = inp_data.to(device)
                output = model(inp_data, target[:-1, :])
                accuracyVal += accuracy(batch, output, onehot=False).item()
                output = output.reshape(-1, output.shape[2]) #keep last dimension
                if onehot:
                    _, targets_Original = target.max(dim=2)
                else:
                    targets_Original= target
                targets_Original = targets_Original[1:].reshape(-1)
                loss_eval = criterion(output, targets_Original)
                lossesCE_eval.append(loss_eval.item()) 
            mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
            accuracyVal = accuracyVal/nval
        out = "epoch: "+str(epoch)+", Train loss CE: " + str(mean_lossCETrain) +  ", Val loss CE: "+ str(mean_lossVal)+ ", errorVal: "+str(accuracyVal)
        
        
        if epoch%200==0:
            model.eval()
            criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.padIndex, reduction='none')
            scoreHungarianVal = HungarianMatchingBS(pds_val, model, 100)
            scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
            scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
            scoreMatchingValClose = sum((scoHVal[0]==scoHVal[1])[maskValclose])
            scoreMatchingValFar = sum((scoHVal[0]==scoHVal[1])[maskValfar])
            out+= ", scoreMatching Val :" +str(scoreMatchingVal)+", scoreMatchingValClose: " +str(scoreMatchingValClose)+", scoreMatchingVal Far: "+ str(scoreMatchingValFar)
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=model_path_save)
        out+="\n"
        f.write(out)
    print("End Training")
    f.close()
    
    

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])