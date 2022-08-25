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
import pandas as pd
import numpy as np
import os

from model import CNN_protein 
from datasetOp import MSA


parser = argparse.ArgumentParser(description="Train Net")

# Hyperparameters
parser.add_argument("--learning_rate", dest="learning_rate", default=0.001, help="Learning Rate")
parser.add_argument("--epochs", dest="epochs", default=5, help="Number of Epochs")
parser.add_argument("--batch_size", dest="batch_size", default=5, help="Batch Size")
parser.add_argument("--depth", dest="depth", default=32, help="Number of Residual Block")
parser.add_argument("--dropout", dest="dropout", default=0.3, help="Dropout Value")

# Path for Train/Validate/Dataset
#Change a False
parser.add_argument("--train", dest="train", default=True, help="Choose for training")
parser.add_argument("--validate", dest="validate", default=True, help="Choose for validate")
parser.add_argument("--dataset", dest="dataset", default='/media/lisa/UNI/ML/training_set_Rosetta/dataset/npz', help="Path to Dataset Elements")

# Comet Specifics
parser.add_argument("--device", dest="device", default='0', help="choose GPU")
parser.add_argument("--project", dest="project", default='CNN_protein', help="Define Comet Project Folder")
parser.add_argument("--experiment", dest="experiment", default='None', help="Define Comet Experiment Name")
parser.add_argument("--note", dest="note", default=None, help="Some additionale note about the experiment")
parser.add_argument("--weights_path", dest="weights_path", default=None, help="Path to the folder with the Model Weights")

args = parser.parse_args()


# Experiment with Comet
experiment = Experiment(api_key="MmTUbWvVERazQRViuiCXkyEFH", project_name=args.project)
experiment.set_name(args.experiment)
#experiment.log_parameters({TODO})

# Device specifications
device = 'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
#print("Device name: ", device, torch.cuda.get_device_name(int(args.device)))


#transforms serve per i dati
if args.train:
    file_csv = '/home/lisa/Desktop/CNN_protein/training_set.csv'
else: 
    file_csv = '/home/lisa/Desktop/CNN_protein/validation_set.csv'

npz = args.dataset
L = 256 # Dimension of matrix 

learning_rate = int(args.learning_rate)
epochs = int(args.epochs)
batch_size = int(args.batch_size)
depth = int(args.depth)
dropout = int(args.dropout)

channels = 82 #da calcolare

# Loading and Transforming data
trainset = MSA(file_csv, npz, L)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=1) #num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)

#inizializzazione rete
net = CNN_protein(channels, L, depth, dropout)

#ottimizzatore e loss
lossFunction = nn.BCELoss() 
optimizer = opt.Adam(net.parameters(), learning_rate) 
#torch.autograd.set_detect_anomaly(True)

net = net.to(device)
#training 
for e in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs = data['msa']
        print(inputs.shape)
        print(inputs.type)
        labels = data['distances']
        optimizer.zero_grad()
        outputs = net(inputs.float())
        
        #backward prop
        loss = lossFunction(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%2000==1999:
            print('[%d, %5d] loss: %.3f'%(epochs+1,i+1,running_loss/2000))
    

