from cgi import print_form
from tracemalloc import start
import numpy as np 
import pandas as pd
import torch
import math
import os
import time

import torch
from torch.utils.data import Dataset


class FixedDimension(object):
    #Prende i primi 256 elementi nelle sequenze del msa
    def __init__(self, size, n_seq): #dimensione della matrice
        self.size = size
        self.n_seq = n_seq

    def get(self, msa):
        if msa.shape[1] >= self.size:
            msa = msa[:, 0:self.size]
        else:
            zero = torch.zeros((msa.shape[0], self.size-msa.shape[1]), dtype=int)
            a = sum(zero, 21)
            msa = torch.cat((msa, a), dim=1)

        if msa.shape[0] > self.n_seq:
            msa = msa[0:self.n_seq, :]
        else:
            zero = torch.zeros((self.n_seq-msa.shape[0], self.size), dtype=int)
            b = sum(zero, 21)
            msa = torch.cat((msa, b), dim=0)
        
        msa = msa.to(int)

        if msa.shape == (self.n_seq, self.size):
            return msa
        else:
            print("Error: Wrong MSA matrix dimension!")
            return []


class OneHotEncoded(object):
    #Genera per ogni aminoacido una hot encoded matrix
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def get(self, msa):
        sequence = msa[0, :]
        print(sequence)
        #sequence = torch.tensor([0, 1,1,1,19,1,1,1,18])

        a = torch.zeros((20, self.size), dtype=int)
        for i in range(20):
            if i in sequence:
                for j in sequence:
                    a[i, j] = (a[i, j] + 1 if j == i else a[i, j])
        print(a)
        print(a.shape)



        '''
        zero = torch.zeros((1), dtype=int)
        one = torch.ones((1), dtype=int)
        for i in range(20):
            if i in sequence: 
                a = ([])
                a = torch.tensor(a, dtype=int)
                for j in sequence:
                    if j == i:
                        a = torch.cat((a, one), dim=0)
                    else:
                        a = torch.cat((a, zero), dim=0)
                sample = torch.cat((sample, a), dim=1)
            else:
                sample = torch.cat((sample, torch.zeros((self.size), dtype=int)), dim=1)
        print(sample.shape)
        sample = torch.reshape(sample, (21, self.size))
        print(sample.shape)
        sample = sample[1:21, :]

        #From row of hot encoded sub array to nparray of shape = (20, self.size, self.size)
        sample = torch.unsqueeze(sample, 0)
        t = sample
        for i in range(self.size-1):
            sample = torch.cat((sample, t), dim=0)

        #From numpy to torch.tensor of shape = (40, self.size, self.size)
        sample = sample.to(int)
        

        if sample.shape == (40, self.size, self.size):
            return sample
        else:
            print("Error: Wrong matrix dimension in hot encoded!")
            return []

        '''

        
        '''print(sequence)
        freq = torch.bincount(sequence, minlength=20)
        print(freq)
        freq = freq[0:20]
        print(freq)
        freq = freq.unsqueeze(0)
        sample = freq
        for i in range()'''


class PSFM(object):
    #Genera Position Specific Frequency Matrix: https://www.researchgate.net/publication/320173501_PSFM-DBT_Identifying_DNA-Binding_Proteins_by_Combing_Position_Specific_Frequency_Matrix_and_Distance-Bigram_Transformation
    #tenedo in considerazione gli MSA
    def __init__(self, size): 
        self.size = size

    def get(self, msa):
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


class ShannonEntropy(object):
    def __init__(self, size): 
        self.size = size

    def get(self, msa):
        entropy = np.zeros(msa.shape[1])
        for j in range(msa.shape[1]):
            for i in range(21):
                freq = np.count_nonzero(msa[:, j] == i) / (msa.shape[0])
                #freq = torch.bincount(sequence, minlength=20)
                if freq > 0:
                    entropy[j] = entropy[j] + -(freq * math.log(freq, 2))
        
        entropy = entropy[:, np.newaxis]
        t = entropy
        for i in range(msa.shape[1]-1):
            entropy = np.concatenate((entropy, t), axis=1)
        
        entropy = entropy[np.newaxis, :, :]
        entropy = torch.from_numpy(entropy)
        entropy = torch.cat((torch.permute(entropy, (0, 2, 1)), entropy), 0)

        if entropy.shape == (2, self.size, self.size):
            return entropy
        else:
            print("Error: Wrong matrix dimension in Shannon Entropy!")
            return []


class CovarianceMatrix(object):
    def __init__(self, size):
        self.size = size

    def get(self, msa):

        #calcolo degli weights 
        weight = np.ones(msa.shape[0])
        for i in range(msa.shape[0]):
            for j in range(i+1, msa.shape[0]):
                similar = int(msa.shape[1] * 0.2)
                for k in range(msa.shape[1]):
                    if similar > 0:
                        if (msa[i, k] != msa[j, k]):
                            similar = similar - 1
                if similar > 0:
                    weight[i] = weight[i] + 1
                    weight[j] = weight[j] + 1
        print(weight)

        #calcolo Meff 
        weight = 1/weight
        meff = np.sum(weight)

        #calcolo frequenza
        pa = np.ones((msa.shape[1], 21)) #Matrice Lx21
        for i in range(msa.shape[1]):
            for a in range(21):
                pa[i, a] = 1.0 
            for k in range(msa.shape[0]):
                a = msa[k, i]
                if a < 21:
                    pa[i, a] = pa[i, a] + weight[k]
            for a in range(21):
                pa[i, a] = pa[i, a] / (21.0 + meff)
                
        pab = np.zeros((msa.shape[1], msa.shape[1], 21, 21)) #Matrice LxLx21x21
        for i in range(msa.shape[1]):
            for j in range(msa.shape[1]):
                for a in range(21):
                    for b in range(21):
                        pab[i, j, a, b] = 1.0 / 21.0
                for k in range(msa.shape[0]):
                    a = msa[k, i]
                    b = msa[k, j]
                    if (a < 21 and b < 21):
                        pab[i, j, a, b] = pab[i, j, a, b] + weight[k]
                for a in range(21):
                    for b in range(21):
                        pab[i, j, a, b] = pab[i, j, a, b] / (21.0 + meff)

        #calcolo matrice di covarianza
        cov = np.zeros((msa.shape[1], msa.shape[1], 21, 21)) #Matrice LxLx21x21
        for a in range(21):
            for b in range(21):
                for i in range(msa.shape[1]):
                    for j in range(msa.shape[1]):
                        cov[i, j, a, b] = pab[i, j, a, b] - (pa[i, a] * pa[j, b])
        cov_final = cov.reshape(msa.shape[1], msa.shape[1], 21*21)
        cov_final = torch.from_numpy(cov_final)
        cov_final = torch.permute(cov_final, (2, 0, 1))

        if cov_final.shape == (441, self.size, self.size):
            return cov_final
        else:
            print("Error: Wrong matrix dimension in Covariance Matrix!")
            return []
        

class Distances(object):
    #Prende i primi 256x256 valori della Contact Map
    def __init__(self, size): 
        self.size = size

    def get(self, dist):
        if dist.shape[0] >= self.size:
            dist = dist[0:self.size, 0:self.size]
        else:
            zero = torch.zeros((dist.shape[0], self.size-dist.shape[1]), dtype=int)
            a = sum(zero, 21)
            dist = torch.cat((dist, a), dim=1)
            zero = torch.zeros((self.size-dist.shape[0], self.size), dtype=int)
            b = sum(zero, 21)
            dist = torch.cat((dist, b), dim=0)

        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if dist[i,j] > 0 and dist[i,j] <= 8:
                    dist[i,j] = 1
                else:
                    dist[i,j] = 0

        if dist.shape == (self.size, self.size):
            #dist = dist.astype(int)
            #dist = torch.from_numpy(dist)
            dist = dist.to(int)
            dist = dist[None, :, :]
            return dist
        else:
            print("Error: Wrong CONTACT MAP dimension!") 
            return []


class MSA(Dataset):

    def __init__(self, file_csv, npz, size, n_seq, device):
        self.file_csv = pd.read_csv(file_csv)
        self.npz = npz
        self.size = size
        self.n_seq = n_seq
        self.device = device

        self.dist = Distances(self.size)

        self.fixed = FixedDimension(self.size, self.n_seq)
        self.ohe = OneHotEncoded(self.size)
        #self.psfm = PSFM(self.size)
        #self.se = ShannonEntropy(self.size)
        #self.cov = CovarianceMatrix(self.size)
        

    def __len__(self):
        return self.file_csv.shape[0]

    def getitem(self, index):
        file = np.load(os.path.join(self.npz, self.file_csv.iloc[index, 1]+'.npz'))
        dist = torch.from_numpy(file['dist6d']).to(self.device)
        msa = torch.from_numpy(file['msa']).to(self.device)
        if self.size > 0:
            # Ground Truth generation
            d = self.dist.get(dist)
            print('Distances DONE')

            # Resize of MSA to SxL
            item = self.fixed.get(msa)
            print('Fixed DONE')
            
            # One Hot Encoded = 40xLxL
            transf1 = self.ohe.get(item)
            print('One Hot Encoded DONE')
            '''
            # PSFM = 42xLxL
            transf2 = self.psfm.get(item)
            print('Position SFM DONE')

            # Shannon = 2xLxL
            transf3 = self.se.get(item)
            print('Shannon DONE')

            # Covariance = 441xLxL
            transf4 = self.cov.get(item)
            print('Covariance DONE')
            '''
            # Features Tensor of ChannelsxLxL TODO
            #msa = torch.cat((transf1, transf2, transf3, transf4), dim=0)
    
            sample = {'msa': transf1, 'distances': d}
            

        return sample






if __name__ == "__main__":
    file_csv = '/home/lisa/Desktop/CNN_protein/training_set.csv'
    npz = '/media/lisa/UNI/ML/training_set_Rosetta/dataset/npz'
    device = 'cpu'
    data = MSA(file_csv, npz, 8, 16, device)
    #print(data)
    e = data.getitem(12)
    print(e)
    print(e['msa'])
    print(e['msa'].shape)
    print(e['msa'].type)
    print(e['distances'].shape)
    print(e['distances'].type)
    #msat = e['msa']
    #msa = e['msa']
    #msad = np.array([[0,0,0,0], [0,0,0,0],[0,0,0,0],[2, 2, 1, 5], [1, 1, 3, 1], [1, 1, 3, 1],[1, 2, 2, 2],[2, 2, 1, 5]])
    
    #print(e)
    #print(msat.shape)
    #print(msa.shape)

    #prova dimensionalità array finale dei sample 

    #inizia codice per generare msa per le operazioni delle matrici di covarianza 
    '''   
    start =time.time()
    weight = np.ones(msa.shape[0])
    for i in range(msa.shape[0]):
        for j in range(i+1, msa.shape[0]):
            similar = int(msa.shape[1] * 0.2)
            for k in range(msa.shape[1]):
                if similar > 0:
                    if (msa[i, k] != msa[j, k]):
                        similar = similar - 1
            if similar > 0:
                weight[i] += 1
                weight[j] += 1
    
    #calcolo Meff 
    weight = 1/weight
    meff = np.sum(weight)

    #calcolo frequenza
    h = time.time()
    pa = np.ones((msa.shape[1], 21)) #Matrice Lx21
    for i in range(msa.shape[1]):
        for a in range(21):
            pa[i, a] = 1.0 
        for k in range(msa.shape[0]):
            a = msa[k, i]
            if a < 21:
                pa[i, a] = pa[i, a] + weight[k]
        for a in range(21):
            pa[i, a] = pa[i, a] / (21.0 + meff)
  
    pab = (np.ones((msa.shape[1], msa.shape[1], 21, 21))) / 21.0 #Matrice LxLx21x21
    for i in range(msa.shape[1]):
        for j in range(msa.shape[1]):
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = 1.0 / 21.0
            for k in range(msa.shape[0]):
                a = msa[k, i]
                b = msa[k, j]
                if (a < 21 and b < 21):
                    pab[i, j, a, b] = pab[i, j, a, b] + weight[k]
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = pab[i, j, a, b] / (21.0 + meff)
    
    
    #calcolo matrice di covarianza
    cov = np.zeros((msa.shape[1], msa.shape[1], 21, 21)) #Matrice LxLx21x21
    for a in range(21):
        for b in range(21):
            for i in range(msa.shape[1]):
                for j in range(msa.shape[1]):
                    cov[i, j, a, b] = pab[i, j, a, b] - (pa[i, a] * pa[j, b])
    cov_final = cov.reshape(msa.shape[1], msa.shape[1], 21*21)
    i = time.time()
    print(i-h)
    
    end = time.time()
    print('OLD:' + str(weight))
    print(end - start)
    pat = pa
    
    f = time.time()
    weight = np.ones(msat.shape[0])
    for i in range(msat.shape[0]):
        for j in range(i+1, msat.shape[0]):
            similar = int(msat.shape[1] * 0.2)
            similar = similar - (sum(msat[i, :] != msat[j, :]))
            if similar > 0:
                weight[i] += 1
                weight[j] += 1

    #calcolo Meff 
    weight = 1/weight
    meff = np.sum(weight)

    #calcolo frequenza
    pa = np.ones((msat.shape[1], 21), dtype=float) #Matrice Lx21
    #for i in range(msat.shape[1]):
    for k in range(msat.shape[0]):
        a = msat[k, :]
        pa[:, a[k]] = (pa[:, a[k]] + weight[k] if a[k] < 21 else pa[:, a[k]])
    for a in range(21):
        pa[:, a] = pa[:, a] / (21.0 + meff)
    
    
    #calcolo matrice di covarianza
    cov = np.zeros((msat.shape[1], msat.shape[1], 21, 21)) #Matrice LxLx21x21
    for a in range(21):
        for b in range(21):
            for i in range(msat.shape[1]):
                for j in range(msat.shape[1]):
                    cov[i, j, a, b] = pab[i, j, a, b] - (pa[i, a] * pa[j, b])
    cov_final = cov.reshape(msat.shape[1], msat.shape[1], 21*21)
    
    g = time.time()
    print('NEW:' + str(weight))
    print(g - f)
    print(sum(pat-pa))
    
    '''
    