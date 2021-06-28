import torch 
import torch.nn as nn 

import torch 
import torch.nn as nn 

class AlexNet(nn.Module):
    
    def __init__(self):

        super(AlexNet,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(11,11),stride=(4,4),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=(1,1))
        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,X):

        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool(X)

        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool(X)

        X = self.conv3(X)
        X = self.relu(X)

        X = self.conv4(X)
        X = self.relu(X)

        X = self.conv5(X)
        X = self.relu(X)
        X = self.pool(X)

        X = X.reshape(X.shape[0],-1)

        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X

X = torch.randn(64,3,227,227)
model = AlexNet()
print(model(X).shape)        
print(model)



