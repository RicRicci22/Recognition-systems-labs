import torch.nn as nn
import torch

class PlainBlock(nn.Module):
    '''
    Implements one block of the plain CNN network described in the paper "Deep residual learning for image recognition" by He et al. (2015)
    '''
    def __init__(self, in_ch, out_ch, kernel_size, num_layers, first_layer_stride=2, padding=1):
        super(PlainBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            if(i==0):
                self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=first_layer_stride, bias=False, padding=padding))
            else:
                self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, bias=False, padding=padding))

            self.layers.append(nn.BatchNorm2d(out_ch))
            self.layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)

class PlainCNN(nn.Module):
    def __init__(self, config, n_classes):
        super(PlainCNN, self).__init__()
        self.config = config

        self.pool = nn.MaxPool2d(2, 2)

        self.block1 = PlainBlock(3, 64, 7, config[0], first_layer_stride=2, padding=3)
        self.block2 = PlainBlock(64, 64, 3, config[1], first_layer_stride=1)
        self.block3 = PlainBlock(64, 128, 3, config[2], first_layer_stride=2)
        self.block4 = PlainBlock(128, 256, 3, config[3], first_layer_stride=2)
        self.block5 = PlainBlock(256, 512, 3, config[4], first_layer_stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,n_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
    def test_forward(self, x):
        '''
        This function is used to test the forward pass of the network. We can use it to check the shapes of the tensors at each layer.
        '''
        print(x.shape)
        x = self.block1(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.block3(x)
        print(x.shape)
        x = self.block4(x)
        print(x.shape)
        x = self.block5(x)
        print(x.shape)
        x = self.avg_pool(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        
    def _getname_(self):
        return "PlainCNN_"+str(sum(self.config)+1)+"_layers"