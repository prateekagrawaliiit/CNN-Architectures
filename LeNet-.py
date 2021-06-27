import torch
import torch.nn as nn 

# http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=(0,0),stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=(0,0),stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),padding=(0,0),stride=(1,1))
        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84,10)

    def forward(self,X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = X.reshape(X.shape[0],-1)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return X


X = torch.randn(64,1,32,32)
model = LeNet()
print(model(X).shape)        
print(model)


