import torch
import torch.nn as nn 


class CNN_protein(nn.Module):
    def __init__(self, channels, residualChannels, dim, depth, dropout):

        super(CNN_protein, self).__init__()
        self.dim = dim
        self.channels = channels
        self.rc = residualChannels
        self.depth = depth
        self.dropout = dropout

        self.batchNorm = nn.BatchNorm2d(self.channels)
        self.batchNormR = nn.BatchNorm2d(self.rc)

        self.conv1 = nn.Conv2d(self.channels, self.rc, 1, padding='same')
        self.residualConv1 = nn.Conv2d(self.rc, self.rc, 3, padding='same')
        self.residualConv2 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=2)
        self.residualConv4 = nn.Conv2d(self.rc, self.rc, 3, padding='same', dilation=4)
        self.conv2 = nn.Conv2d(self.rc, 1, 3, padding='same')

        self.drop = nn.Dropout2d(self.dropout)

        self.relu = nn.ReLU()

        self.sig = nn.Sigmoid()


    def residualBlock(self, input, i):
        x = self.batchNormR(input)
        x = self.relu(x)
        x = self.residualConv1(x)
        x = self.drop(x)
        x = self.relu(x)
        if i%3 == 0:
            x = self.residualConv1(x)
        elif i%3 == 1:
            x = self.residualConv2(x)
        else:
            x = self.residualConv4(x)
        return x


    def forward(self, input):
        x = self.batchNorm(input)
        x = self.relu(x)
        x = self.conv1(x)

        #Bias thing: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

        #Max pooling??

        #Residual Block executed depth time
        for i in range(self.depth):
            x = x + self.residualBlock(x, i)
        
        x = self.batchNormR(x)
        x = self.relu(x)
        x = self.conv2(x)
        #nn.Sigmoid?
        output = self.sig(x)

        #simmetrizza?
        return output
    