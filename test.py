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
import csv

from model import CNN_protein, DilatedWithDropout, DilatedResidual, ResidualWithDropout, ResidualBlock
from datasetOp import MSA
from precision import precision



parser = argparse.ArgumentParser(description="CNN Net")

# Hyperparameters
parser.add_argument("--dim", dest="dim", default=256, help="Sequence Dimension")
parser.add_argument("--nseq", dest="nseq", default=1024, help="Number of MSA Sequences")
parser.add_argument("--batch_size", dest="batch_size", default=1, help="Batch Size")
parser.add_argument("--rc", dest="residual_channels", default=64, help="Channel in the Residual Block")
parser.add_argument("--depth", dest="depth", default=32, help="Number of Residual Block")
parser.add_argument("--dropout", dest="dropout", default=0.3, help="Dropout Value")

parser.add_argument("--testset", dest="testset", default='../Dataset/test_set/npz', help="Path to Testset Elements")

# Comet Specifics
parser.add_argument("--device", dest="device", default='0', help="Choose GPU")
parser.add_argument("--project", dest="project", default='CNN_protein', help="Define Comet Project Folder")
parser.add_argument("--experiment", dest="experiment", default='None', help="Define Comet Experiment Name")
parser.add_argument("--note", dest="note", default=None, help="Some additionale note about the experiment")

# Path of weights
parser.add_argument("--weights_path", dest="weights_path", default='SeqCov_00/w_end.pth', help="Path to the folder with the Model Weights")

# Plots
parser.add_argument("--cmap", dest="cmap", default=False, help="Contact Map with TopL")

args = parser.parse_args()


# Experiment with Comet
experiment = Experiment(api_key="MmTUbWvVERazQRViuiCXkyEFH", project_name=args.project)
experiment.set_name(args.experiment)

# Device specifications
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

npz = args.testset
L = int(args.dim)
n_seq = int(args.nseq)

batch_size = int(args.batch_size)
residualChannels = int(args.residual_channels)
depth = int(args.depth)
dropout = float(args.dropout)

channels = 525 

# Loading and Transforming data
test_csv = '../Dataset/test_set/test_set_all.csv'

testset = MSA(test_csv, npz, L, n_seq)
testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

hyper_parameters = {
    'DIMENSION': L,
    'N_SEQUENCES': n_seq,
    'BATCH_SIZE': batch_size,
    'DROPOUT_VALUE': dropout,
    'DEPTH': depth,
    'CHANNELS': channels,
    'RESIDUAL_CHANNELS': residualChannels
}
experiment.log_parameters(hyper_parameters)

#Weights
wpath = './model_weights'
w_path = os.path.join(wpath, args.weights_path)

net = CNN_protein(DilatedWithDropout, depth, channels, residualChannels, L, dropout)
net.load_state_dict(torch.load(w_path, map_location=device))

net = net.to(device)

print(f'START TESTING:')
net.eval() 
lrl5 = 0.0
lrl2 = 0.0
lrl = 0.0
mlrl5 = 0.0
mlrl2 = 0.0
mlrl = 0.0

csv_list = []

with torch.no_grad():
    with tqdm(testloader, unit='batch') as vep:
        for vi, vdata in enumerate(vep):
            vep.set_description(f'Test: ')
            vinput = vdata['msa'].to(device)
            vlabels = vdata['distances'].to(device)

            voutputs = net(vinput.float()).to(device)

            voutputs = torch.squeeze(voutputs)
            voutputs = (voutputs + voutputs.transpose(-1, -2))/2.0
            vlabels = torch.squeeze(vlabels)
            
            if args.cmap:
                vacc = precision(voutputs, vlabels, vdata['slen'], vdata['name'])
            else:
                vacc = precision(voutputs, vlabels, vdata['slen'])
            
            lrl5 += vacc['LR_L/5']
            lrl2 += vacc['LR_L/2']
            lrl += vacc['LR_L']
            mlrl5 += vacc['MLR_L/5']
            mlrl2 += vacc['MLR_L/2']
            mlrl += vacc['MLR_L']
            
            csv_list.append((str(vdata['name']), str(vacc['LR_L']), str(vacc['LR_L/5'])))

experiment.log_metric('VAL_PRECISION_LR_L/5: ', lrl5/len(testloader))
experiment.log_metric('VAL_PRECISION_LR_L/2: ', lrl2/len(testloader))
experiment.log_metric('VAL_PRECISION_LR_L: ', lrl/len(testloader))
experiment.log_metric('VAL_PRECISION_MLR_L/5: ', mlrl5/len(testloader))
experiment.log_metric('VAL_PRECISION_MLR_L/2: ', mlrl2/len(testloader))
experiment.log_metric('VAL_PRECISION_MLR_L: ', mlrl/len(testloader))

with open('../Dataset/test_set/CNN_precision.csv', 'w') as ts:
        ts_writer = csv.writer(ts, delimiter=',', quotechar='"')
        ts_writer.writerow(['NAME', 'L', 'L/5'])
        for i in range(len(csv_list)):
            ts_writer.writerow(csv_list[i])
print(f'END TEST')   