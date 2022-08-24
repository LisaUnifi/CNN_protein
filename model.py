import torch
import torch.nn as nn 
import torch.nn.functional as F 


class Net(nn.Module):
    def __init__(self, channels, dim, depth, dropout):

        super(Net, self).__init__()
        self.dim = dim
        self.depth = depth

        self.batchNorm = nn.BatchNorm2d(channels)
        self.batchNorm64 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(channels, 64, 1)
        self.residualConv1 = nn.Conv2d(64, 64, 3)
        self.residualConv2 = nn.Conv2d(64, 64, 3, dilation=2)
        self.residualConv4 = nn.Conv2d(64, 64, 3, dilation=4)
        self.conv2 = nn.Conv2d(64, 1, 3)

        self.drop = nn.Dropout2d(dropout)


    def residualBlock(self, input, i):
        x = self.batchNorm64(input)
        x = F.relu(x)
        x = self.residualConv1(x)
        x = self.drop(x)
        x = F.relu(x)
        if i%3 == 0:
            x = self.residualConv1(x)
        elif i%3 == 1:
            x = self.residualConv2(x)
        else:
            x = self.residualConv4(x)
        return x


    def forward(self, input):
        #devo definire la matrice?

        x = self.batchNorm(input)
        x = F.relu(x)
        x = self.conv1(x)

        #Residual Block eseguito depth volte
        for i in range(self.depth):
            x += self.residualBlock(x, i)
        
        x = self.batchNorm64(x)
        x = F.relu(x)
        x = self.conv2(x)
        output = F.sigmoid(x)
        return output

    #serve dim??
    