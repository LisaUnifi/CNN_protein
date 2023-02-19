import torch
import torch.nn as nn 

'''
CNN_protein with depth = 16:
==========================================================================================
├─BatchNorm2d: 1-1                       [-1, 525, 256, 256]       1,050
├─ReLU: 1-2                              [-1, 525, 256, 256]       --
├─Conv2d: 1-3                            [-1, 64, 256, 256]        33,664
├─Sequential: 1-4                        [-1, 64, 256, 256]        --
|    └─ResidualBlock: 2-1                [-1, 64, 256, 256]        --
|    |    └─BatchNorm2d: 3-1             [-1, 64, 256, 256]        128
|    |    └─ReLU: 3-2                    [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-3                  [-1, 64, 256, 256]        36,928
|    |    └─Dropout2d: 3-4               [-1, 64, 256, 256]        --
|    |    └─ReLU: 3-5                    [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-6                  [-1, 64, 256, 256]        36,928
|    └─ResidualBlock: 2-2                [-1, 64, 256, 256]        --
                .
                .
                .
|    └─ResidualBlock: 2-16               [-1, 64, 256, 256]        --
|    |    └─BatchNorm2d: 3-91            [-1, 64, 256, 256]        128
|    |    └─ReLU: 3-92                   [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-93                 [-1, 64, 256, 256]        36,928
|    |    └─Dropout2d: 3-94              [-1, 64, 256, 256]        --
|    |    └─ReLU: 3-95                   [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-96                 [-1, 64, 256, 256]        36,928
├─BatchNorm2d: 1-5                       [-1, 64, 256, 256]        128
├─ReLU: 1-6                              [-1, 64, 256, 256]        --
├─Conv2d: 1-7                            [-1, 1, 256, 256]         577
├─Sigmoid: 1-8                           [-1, 1, 256, 256]         --
==========================================================================================
'''


class DilatedWithDropout(nn.Module):
    def __init__(self, residualChannels, dropout, dilation):
        super(DilatedWithDropout, self).__init__()
        self.rc = residualChannels
        self.dropout = dropout
        self.dilation = dilation

        self.batchNormR = nn.BatchNorm2d(self.rc)
        self.conv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.residualConv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.residualConv2 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=2)
        self.residualConv4 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=4)
        self.drop = nn.Dropout2d(self.dropout)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        residual = x
        out = self.batchNormR(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.drop(out)
        out = self.relu(out)
        if self.dilation%3 == 0:
            out = self.residualConv1(out)
        elif self.dilation%3 == 1:
            out = self.residualConv2(out)
        else:
            out = self.residualConv4(out)
        out += residual
        return out


class DilatedResidual(nn.Module):
    def __init__(self, residualChannels, dropout, dilation):
        super(DilatedResidual, self).__init__()
        self.rc = residualChannels
        self.dilation = dilation

        self.batchNormR = nn.BatchNorm2d(self.rc)
        self.conv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.residualConv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.residualConv2 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=2)
        self.residualConv4 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=4)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        residual = x
        out = self.batchNormR(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batchNormR(x)
        out = self.relu(out)
        if self.dilation%3 == 0:
            out = self.residualConv1(out)
        elif self.dilation%3 == 1:
            out = self.residualConv2(out)
        else:
            out = self.residualConv4(out)
        out += residual
        return out


class ResidualWithDropout(nn.Module):
    def __init__(self, residualChannels, dropout, dilation):
        super(ResidualWithDropout, self).__init__()
        self.rc = residualChannels
        self.dropout = dropout
        self.dilation = dilation

        self.batchNormR = nn.BatchNorm2d(self.rc)
        self.conv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.conv2 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.drop = nn.Dropout2d(self.dropout)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        residual = x
        out = self.batchNormR(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ResidualBlock(nn.Module):
    def __init__(self, residualChannels, dropout, dilation):
        super(ResidualBlock, self).__init__()
        self.rc = residualChannels
        self.dropout = dropout
        self.dilation = dilation

        self.batchNormR = nn.BatchNorm2d(self.rc)
        self.conv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.conv2 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.relu = nn.ReLU()

        
    def forward(self, x):
        residual = x
        out = self.batchNormR(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batchNormR(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class CNN_protein(nn.Module):
    def __init__(self, block, depth, channels, residualChannels, dim, dropout):

        super(CNN_protein, self).__init__()
        self.dim = dim
        self.channels = channels
        self.rc = residualChannels
        self.depth = depth
        self.dropout = dropout
        self.block = block

        self.batchNorm = nn.BatchNorm2d(self.channels)
        self.batchNormR = nn.BatchNorm2d(self.rc)
        self.conv1 = nn.Conv2d(self.channels, self.rc, 1, padding='same')
        self.r_block = self._make_layer(block)
        self.conv2 = nn.Conv2d(self.rc, 1, 3, padding='same')
        self.drop = nn.Dropout2d(self.dropout)
        self.relu = nn.ReLU()

        self.sig = nn.Sigmoid()


    def _make_layer(self, block):
        layers = []
        for i in range(1, self.depth+1):
            layers.append(block(self.rc, self.dropout, i))
        return nn.Sequential(*layers)


    def forward(self, input):
        x = self.batchNorm(input)
        x = self.relu(x)
        x = self.conv1(x)

        #Residual Block executed depth time
        x = self.r_block(x)
        
        x = self.batchNormR(x)
        x = self.relu(x)
        x = self.conv2(x)
        output = self.sig(x)

        return output

