import torch
import torch.nn as nn
torch.cuda.empty_cache()

VGG_types = {
"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
"VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"]
}

class VGGNet(nn.Module):

    def __init__(self,in_channels=3,num_classes=1000,arch="VGG16"):

        super(VGGNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_arch(VGG_types[arch])
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
            )


    def forward(self,X):
        X = self.conv_layers(X)
        X = X.reshape(X.shape[0],-1)
        X = self.fcs(X)
        return X

    def create_conv_arch(self,architecture):
        
        layers=[]
        in_channels = self.in_channels

        for i in architecture:

            if type(i) == int:
                out_channels = i
                layers+= [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=(1,1),stride=(1,1)),nn.BatchNorm2d(i),nn.ReLU()]
                in_channels=i

            elif i == 'M':
                layers+= [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        
        return nn.Sequential(*layers)


X = torch.randn(64,3,224,224)
model = VGGNet(in_channels=3,num_classes=1000,arch="VGG19")
print(model(X).shape)        
print(model)
