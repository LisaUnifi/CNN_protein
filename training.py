from comet_ml import Experiment
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import argparse
from tqdm import tqdm
import time
import json
import pandas as pd
import numpy as np
import os

from model import CNN_protein, DilatedWithDropout, DilatedResidual, ResidualWithDropout, ResidualBlock
from datasetOp import MSA
from precision import precision



parser = argparse.ArgumentParser(description="CNN Net")

# Hyperparameters
parser.add_argument("--dim", dest="dim", default=256, help="Sequence Dimension")
parser.add_argument("--nseq", dest="nseq", default=1024, help="Number of MSA Sequences")
parser.add_argument("--learning_rate", dest="learning_rate", default=0.0001, help="Learning Rate")
parser.add_argument("--epochs", dest="epochs", default=5, help="Number of Epochs")
parser.add_argument("--batch_size", dest="batch_size", default=5, help="Batch Size")
parser.add_argument("--rc", dest="residual_channels", default=64, help="Channel in the Residual Block")
parser.add_argument("--depth", dest="depth", default=32, help="Number of Residual Block")
parser.add_argument("--dropout", dest="dropout", default=0.3, help="Dropout Value")

# Path for Train/Validate/Dataset
parser.add_argument("--train", dest="train", default=True, help="Choose TRUE for training+validation and FALSE for test - default TRUE")
parser.add_argument("--dataset", dest="dataset", default='../Dataset/training_set/npz', help="Path to Dataset Elements")
parser.add_argument("--testset", dest="testset", default='../Dataset/test_set/npz', help="Path to Testset Elements")

# Comet Specifics
parser.add_argument("--device", dest="device", default='0', help="choose GPU")
parser.add_argument("--project", dest="project", default='CNN_protein', help="Define Comet Project Folder")
parser.add_argument("--experiment", dest="experiment", default='None', help="Define Comet Experiment Name")
parser.add_argument("--note", dest="note", default=None, help="Some additionale note about the experiment")

args = parser.parse_args()


# Experiment with Comet
experiment = Experiment(api_key="MmTUbWvVERazQRViuiCXkyEFH", project_name=args.project)
experiment.set_name(args.experiment)

# Device specifications
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

npz = args.dataset
t_npz = args.testset
L = int(args.dim)
n_seq = int(args.nseq)

learning_rate = float(args.learning_rate)
epochs = int(args.epochs)
batch_size = int(args.batch_size)
residualChannels = int(args.residual_channels)
depth = int(args.depth)
dropout = float(args.dropout)

channels = 525 

# Loading and Transforming data
if args.train:
    file_csv = './training_set.csv'
    val_csv = './validation_set.csv'

    trainset = MSA(file_csv, npz, L, n_seq)
    valset = MSA(val_csv, npz, L, n_seq)

    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8, pin_memory=True) 
    valloader = DataLoader(valset, 1, shuffle=True, num_workers=8, pin_memory=True)

#inizializzazione rete
net = CNN_protein(DilatedWithDropout, depth, channels, residualChannels, L, dropout)

hyper_parameters = {
    'DIMENSION': L,
    'N_SEQUENCES': n_seq,
    'BATCH_SIZE': batch_size,
    'EPOCH': epochs,
    'LEARNING_RATE': learning_rate,
    'DROPOUT_VALUE': dropout,
    'DEPTH': depth,
    'CHANNELS': channels,
    'RESIDUAL_CHANNELS': residualChannels
}
experiment.log_parameters(hyper_parameters)
experiment.set_model_graph(net, overwrite=True)

#Weights
wpath = './model_weights'
new_path = os.path.join(wpath, args.experiment)

if not os.path.exists(new_path):
    os.makedirs(new_path)

#Hyperparameters dictionary
with open(new_path + '/hp.json', 'w') as hp:
    json.dump(hyper_parameters, hp, indent=4)

#ottimizzatore e loss
lossFunction = nn.BCELoss() 
optimizer = opt.Adam(net.parameters(), learning_rate) 
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 5, threshold = 1e-3)
torch.autograd.set_detect_anomaly(True)

net = net.to(device)

# Parameters counter
tot_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
experiment.log_other('N_PARAMETERS', tot_parameters)
print("Total number of parameters: ", tot_parameters) 

print('START TRAINING:')
for e in range(epochs):
    net.train()
    train_loss = 0.0
    train_precision = 0.0
    with tqdm(trainloader, unit='batch') as ep:
        for i, data in enumerate(ep):
            ep.set_description(f'Epoch {e}: ')
            inputs = data['msa'].to(device)
            labels = data['distances'].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs.float()).to(device)
            
            #backward prop
            loss = lossFunction(outputs, labels.float())
            outputs = torch.squeeze(outputs)
            outputs = (outputs + outputs.transpose(-1, -2))/2.0
            labels = torch.squeeze(labels)
            acc = precision(outputs, labels, L)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_precision += acc['LR_L/5']
        time.sleep(0.1)
    print('Loss: ' + str(train_loss/len(trainloader)))
    print('Precision: ' + str(train_precision/len(trainloader)))
    experiment.log_metric('TRAIN_LOSS: ', train_loss/len(trainloader), step = e + 1)
    experiment.log_metric('TRAIN_PRECISION: ', train_precision/len(trainloader), step = e + 1)
    print(f'END TRAINING EPOCH {e}:')

    if e % 2 == 1:
        print(f'START EVALUATION:')
        net.eval() 
        val_loss = 0.0
        lrl5 = 0.0
        lrl2 = 0.0
        lrl = 0.0
        mlrl5 = 0.0
        mlrl2 = 0.0
        mlrl = 0.0
        with torch.no_grad():
            with tqdm(valloader, unit='batch') as vep:
                for vi, vdata in enumerate(vep):
                    vep.set_description(f'Validation Epoch {e}: ')
                    vinput = vdata['msa'].to(device)
                    vlabels = vdata['distances'].to(device)

                    voutputs = net(vinput.float()).to(device)

                    voutputs = torch.squeeze(voutputs)
                    voutputs = (voutputs + voutputs.transpose(-1, -2))/2.0
                    vlabels = torch.squeeze(vlabels)

                    vloss = lossFunction(voutputs, vlabels.float())
                    val_loss += vloss.item()

                    vacc = precision(voutputs, vlabels, vdata['slen'])
                    lrl5 += vacc['LR_L/5']
                    lrl2 += vacc['LR_L/2']
                    lrl += vacc['LR_L']
                    mlrl5 += vacc['MLR_L/5']
                    mlrl2 += vacc['MLR_L/2']
                    mlrl += vacc['MLR_L']
        print('Evaluation Loss: ' + str(val_loss/len(valloader)))
        scheduler.step(val_loss/len(valloader))
        experiment.log_metric('VAL_LOSS: ', val_loss/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_LR_L/5: ', lrl5/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_LR_L/2: ', lrl2/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_LR_L: ', lrl/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_MLR_L/5: ', mlrl5/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_MLR_L/2: ', mlrl2/len(valloader), step = e + 1)
        experiment.log_metric('VAL_PRECISION_MLR_L: ', mlrl/len(valloader), step = e + 1)
        print(f'END EVALUATION EPOCH {e}:')       
    
    if e % 2 == 1:
        torch.save(net.state_dict(), new_path + '/w_' + str(e + 1) + '.pth')
torch.save(net.state_dict(), new_path + '/w_end' + '.pth')

print(f'END TRAINING!')
experiment.end()

        