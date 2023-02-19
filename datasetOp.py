import covariance
import torch
from torch.utils.data import Dataset
import math
import pandas as pd
import numpy as np
import pickle
import os


class FixedDimension(object):
    def __init__(self, size, n_seq): #dimensione della matrice
        self.size = size
        self.n_seq = n_seq

    def __call__(self, msa):
        if msa.shape[1] >= self.size:
            msa = msa[:, 0:self.size]
            slen = self.size
        else:
            slen = msa.shape[1]
            zero = np.zeros((msa.shape[0], self.size-msa.shape[1]), dtype=int)
            a = 21 + zero
            msa = np.concatenate((msa, a), axis=1)

        if msa.shape[0] > self.n_seq:
            msa = msa[0:self.n_seq, :]
        else:
            zero = np.zeros((self.n_seq-msa.shape[0], self.size), dtype=int)
            b = 21 + zero
            msa = np.concatenate((msa, b), axis=0)
        
        msa = msa.astype(int)

        if msa.shape == (self.n_seq, self.size):
            return msa, slen
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
        for i in range(20):
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
    def __init__(self, size): 
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

        #To nparray of shape = (21, self.size, self.size)
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

    def __call__(self, msa):
        entropy = np.zeros(msa.shape[1])
        for j in range(msa.shape[1]):
            for i in range(21):
                freq = np.count_nonzero(msa[:, j] == i) / (msa.shape[0])
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

    def __call__(self, msa):

        #Cij^AB = fij(A,B) - fi(A)fj(B)
        
        f_msa = np.array(msa, dtype=float)
        cov_final = covariance.with_cython(f_msa)
        cov_final = torch.from_numpy(cov_final)
        cov_final = torch.permute(cov_final, (2, 0, 1))

        if cov_final.shape == (441, self.size, self.size):
            return cov_final
        else:
            print("Error: Wrong matrix dimension in Covariance Matrix!")
            return []
        

class Distances(object):
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

        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if dist[i,j] > 0 and dist[i,j] <= 8:
                    dist[i,j] = 1
                else:
                    dist[i,j] = 0

        if dist.shape == (self.size, self.size):
            dist = dist.astype(int)
            dist = torch.from_numpy(dist)
            dist = dist[None, :, :]
            return dist
        else:
            print("Error: Wrong CONTACT MAP dimension!") 
            return []


class MSA(Dataset):

    def __init__(self, file_csv, npz, size, n_seq):
        self.file_csv = pd.read_csv(file_csv)
        self.npz = npz
        self.size = size
        self.n_seq = n_seq

        self.dist = Distances(self.size)

        self.fixed = FixedDimension(self.size, self.n_seq)
        self.ohe = OneHotEncoded(self.size)
        self.psfm = PSFM(self.size)
        self.se = ShannonEntropy(self.size)
        self.cov = CovarianceMatrix(self.size)

    def __len__(self):
        return self.file_csv.shape[0]

    def __getitem__(self, index):
        file = np.load(os.path.join(self.npz, self.file_csv.iloc[index, 1]+'.npz'))
        name = self.file_csv.iloc[index, 1]
        if self.size > 0:
            # Ground Truth generation
            d = self.dist(file['dist6d'])

            # Resize of MSA to SxL
            item, slen = self.fixed(file['msa'])

            # One Hot Encoded = 40xLxL
            transf1 = self.ohe(item)

            # PSFM = 42xLxL
            transf2 = self.psfm(item)

            # Shannon = 2xLxL
            transf3 = self.se(item)

            # Covariance = 441xLxL
            transf4 = self.cov(item)

            # Features Tensor of 525xLxL 
            #msa = torch.cat((transf1, transf2, transf3), dim=0)
            msa = torch.cat((transf1, transf2, transf3, transf4), dim=0)
            #sample = {'msa': transf4, 'distances': d}
            sample = {'msa': msa, 'distances': d, 'name': name, 'slen': slen}

        return sample


