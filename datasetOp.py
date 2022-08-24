import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import numpy as np
import os


class FixedDimension(object):
    #Prende i primi 256 elementi nelle sequenze del msa
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def __call__(self, msa):
        if msa.shape[1] >= self.size:
            msa = msa[:, 0:self.size]
        else:
            zero = np.zeros((msa.shape[0], self.size-msa.shape[1]), dtype=int)
            a = 21 + zero
            msa = np.concatenate((msa, a), axis=1)
        
        msa = msa.astype(int)
        msa = torch.from_numpy(msa)

        if msa.shape[1] == self.size:
            return msa
        else:
            print("Error: Wrong MSA matrix dimension!")
            return []


class OneHotEncoded(object):
    #Genera per ogni aminoacido una hot encoded matrix
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def __call__(self, msa):
        sequence = msa[0, :]
        sample = sequence
        for i in range(self.size):
            if i in sequence: 
                a = ([])
                for j in sequence:
                    if j == i:
                        a = np.append(a, 1)
                    else:
                        a = np.append(a, 0)
                sample = np.append(sample, a, axis=0)
            else:
                sample = np.append(sample, np.zeros((self.size), dtype=int), axis=0)
        sample = sample.reshape(21, self.size)
        sample = sample[1:21, :]

        #From row of hot encoded sub array to nparray of shape = (20, self.size, self.size)
        sample = sample[:, :, np.newaxis]
        t = sample
        for i in range(self.size-1):
            sample = np.concatenate((sample, t), axis=2)

        #From numpy to torch.tensor of shape = (40, self.size, self.size)
        sample = sample.astype(int)
        sample = torch.from_numpy(sample)
        sample = torch.cat((torch.permute(sample, (0, 2, 1)), sample), 0)

        if sample.shape == (40, self.size, self.size):
            return sample
        else:
            print("Error: Wrong matrix dimension in hot encoded!")
            return []


class PSFM(object):
    #Genera Position Specific Frequency Matrix: https://www.researchgate.net/publication/320173501_PSFM-DBT_Identifying_DNA-Binding_Proteins_by_Combing_Position_Specific_Frequency_Matrix_and_Distance-Bigram_Transformation
    #tenedo in considerazione gli MSA
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def __call__(self, msa):
        sample = np.zeros((self.size), dtype=int)
        for i in range(21):
            index = np.zeros((self.size), dtype=int)
            for j in msa:
                for z in range(len(j)):
                    if j[z] == i:
                        index[z] += 1
            index = np.divide(index, msa.shape[0])
            sample = np.append(sample, index, axis=0)
        sample = sample.reshape(22, self.size)
        sample = sample[1:22, :]

        #From row of hot encoded sub array to nparray of shape = (21, self.size, self.size)
        sample = sample[:, :, np.newaxis]
        t = sample
        for i in range(self.size-1):
            sample = np.concatenate((sample, t), axis=2)

        #From numpy to torch.tensor of shape = (42, self.size, self.size)
        sample = torch.from_numpy(sample)
        sample = torch.cat((torch.permute(sample, (0, 2, 1)), sample), 0)

        if sample.shape == (42, self.size, self.size):
            return sample
        else:
            print("Error: Wrong matrix dimension in PSFM!")
            return []



class Distances(object):
    #Prende i primi 256x256 valori della Contact Map
    def __init__(self, size): 
        self.size = size

    def __call__(self, dist):
        if dist.shape[0] >= self.size:
            dist = dist[0:self.size, 0:self.size]
        else:
            zero = np.zeros((dist.shape[0], self.size-dist.shape[1]), dtype=int)
            a = 21 + zero
            dist = np.concatenate((dist, a), axis=1)
            zero = np.zeros((self.size-dist.shape[0], self.size), dtype=int)
            b = 21 + zero
            dist = np.concatenate((dist, b), axis=0)

        if dist.shape == (self.size, self.size):
            dist = torch.from_numpy(dist)
            return dist
        else:
            print("Error: Wrong CONTACT MAP dimension!") #Giusto?
            return []


class MSA(Dataset):

    # TODO: Ricorda di trasformare tutto in un tensore torch

    #quando viene inizializzato l'oggetto del dataset
    def __init__(self, file_csv, npz, size):
        self.file_csv = pd.read_csv(file_csv)
        self.npz = npz
        self.size = size

        self.dist = Distances(self.size)

        self.fixed = FixedDimension(self.size)
        self.ohe = OneHotEncoded(self.size)
        self.psfm = PSFM(self.size)
        self.shannon 

    #ritorna quanti samples ci sono nel dataset
    def __len__(self):
        return self.file.shape[0]

    #ritorna un sample con un dato indice e trasformazione quando presente 
    def __getitem__(self, index):
        file = np.load(os.path.join(self.npz, self.file_csv.iloc[index, 1]+'.npz'))
        #item = {'msa': file['msa'], 'distances': file['dist6d']}
        if self.size > 0:
            d = self.dist(file['dist6d'])
            item = self.fixed(file['msa'])
            transf1 = self.ohe(item)
            transf2 = self.psfm(item)

            sample = {'msa': file['msa'], 'distances': d}
        return sample

