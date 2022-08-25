import numpy as np 
import pandas as pd
import torch
import os

class FixedDimension(object):
    #Prende i primi 256 elementi nelle sequenze del msa
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def get(self, msa):
        if msa.shape[1] >= self.size:
            msa = msa[:, 0:self.size]
        else:
            zero = np.zeros((msa.shape[0], self.size-msa.shape[1]), dtype=int)
            a = 21 + zero
            msa = np.concatenate((msa, a), axis=1)
        
        msa = msa.astype(int)
        #msa = torch.from_numpy(msa)

        if msa.shape[1] == self.size:
            print('Fixed Done')
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
        print('SEQUENCE: ' + str(sequence.shape))
        sample = sequence
        print(sample)
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
            print('OHE Done')
            return sample
        else:
            print("Error: Wrong matrix dimension in hot encoded!")
            return []


class PSFM(object):
    #Genera Position Specific Frequency Matrix: https://www.researchgate.net/publication/320173501_PSFM-DBT_Identifying_DNA-Binding_Proteins_by_Combing_Position_Specific_Frequency_Matrix_and_Distance-Bigram_Transformation
    #tenedo in considerazione gli MSA
    def __init__(self, size): #dimensione della matrice
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
            print('PSFM Done')
            return sample
        else:
            print("Error: Wrong matrix dimension in PSFM!")
            return []


class ShannonEntropy(object):
    #Genera Position Specific Frequency Matrix: https://www.researchgate.net/publication/320173501_PSFM-DBT_Identifying_DNA-Binding_Proteins_by_Combing_Position_Specific_Frequency_Matrix_and_Distance-Bigram_Transformation
    #tenedo in considerazione gli MSA
    def __init__(self, size): #dimensione della matrice
        self.size = size

    def __call__(self, msa):
        sample = np.zeros((self.size), dtype=int)

        if sample.shape == (42, self.size, self.size):
            return sample
        else:
            print("Error: Wrong matrix dimension in PSFM!")
            return []
    
    def shannon_entropy_corrected(dna_sequence):
        """Custom implementation of shannon entropy with a full non-binarized sequence
            Formula looks like this
            H(S) = −Σ P(Si) log2 (P(Si))
            P(Si) here is simply the relative frequency of character A,T,G,C or n in the string.
        """
        entropy = 0
        for nucleotide in {'A', 'T', 'G', 'C', 'N'}:
            rel_freq = dna_sequence.count(nucleotide) / len(dna_sequence)
            if rel_freq > 0:
                entropy = entropy + -(rel_freq * math.log(rel_freq, 2))
            
        return entropy


class Distances(object):
    #Prende i primi 256x256 valori della Contact Map
    def __init__(self, size): 
        self.size = size

    def get(self, dist):
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
            return dist
        else:
            print("Error: Wrong CONTACT MAP dimension!") #Giusto?
            return []


class MSA(object):

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
        #self.shannon 

    #ritorna quanti samples ci sono nel dataset
    def len(self):
        return self.file_csv.shape[0]

    #ritorna un sample con un dato indice e trasformazione quando presente 
    def getitem(self, index):
        file = np.load(os.path.join(self.npz, self.file_csv.iloc[index, 1]+'.npz'))
        #item = {'msa': file['msa'], 'distances': file['dist6d']}
        if self.size > 0:
            d = self.dist.get(file['dist6d'])
            item = self.fixed.get(file['msa'])
            transf1 = self.ohe.get(item)
            transf2 = self.psfm.get(item)
            transf1 = torch.permute(transf1, (1, 2, 0))
            transf2 = torch.permute(transf2, (1, 2, 0))
            msa = torch.cat((transf1, transf2), dim=2)
            msa = torch.permute(msa, (2, 0, 1))
            sample = {'msa': msa, 'distances': d}
            print(sample['msa'].shape)
        return sample



if __name__ == "__main__":
    file_csv = '/home/lisa/Desktop/CNN_protein/training_set.csv'
    npz = '/media/lisa/UNI/ML/training_set_Rosetta/dataset/npz'
    #data = MSA(file_csv, npz, 256)
    #print(data)

    #print(data.getitem(12))

    x = torch.tensor([[[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]],[[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]]])
    print(x.shape)

    x = x[:, None, :, :]
    print(x.shape)

    '''
    for i in range(20):
        if i in sequence: 
            a = ([])
            for j in sequence:
                if j == i:
                    a = np.append(a, 1)
                else:
                    a = np.append(a, 0)
                print(a)
            sample = np.append(sample, a, axis=0)
        else:
            sample = np.append(sample, np.zeros((6), dtype=int), axis=0)
    sample = sample.reshape(21, 6)
    sample = sample[1:21, :]
    print(sample)
    sample = sample[:, :, np.newaxis]
    t = sample
    print(sample.shape)
    for i in range(6-1):
        sample = np.concatenate((sample, t), axis=2)
        #print('Iterazione ' + str(i))
        #print(sample)
    #sample = sample.reshape(20, 6, 6)
    print(sample.shape)
    print(sample)

    sample = sample.astype(int)
    sample = torch.from_numpy(sample)
    sample = torch.cat((torch.permute(sample, (0, 2, 1)), sample), 0)
    print(sample.shape)
    print(sample)
    '''



    #dist = file['dist6d']
    #print(dist.shape)
    '''
    txt = '/home/lisa/Desktop/ML/training_set.csv'
    df = pd.read_csv(txt)
    print(df.shape[0])
    for i in range(df.shape[0]):
        print(df.iloc[i,1])
    print(df.iloc[0,1])

    index = [3, 5, 88]
    npz = '/media/lisa/UNI/ML/training_set_Rosetta/dataset/npz'
    protein = []
    for i in index:
        protein.append(os.path.join(npz, df.iloc[i, 1]+'.npz'))
    print(protein)
    '''