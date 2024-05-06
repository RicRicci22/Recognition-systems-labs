import torch.nn as nn
import torch

# PLAIN CNN
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
    
# SKIP CNN
class SkipLayer(nn.Module):
    '''
    Implements a single skip connection sub-block of the residual CNN. It consists of two convolutional layers with batch normalization and ReLU activation.
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride_first_conv=1):
        super(SkipLayer, self).__init__()
        layers = []
        self.stride_first_conv = stride_first_conv
        if(stride_first_conv!=1):
            self.conv1x1 = nn.Conv2d(in_ch, in_ch*2, 1, stride=stride_first_conv, bias=False)
            self.bn = nn.BatchNorm2d(in_ch*2)
        
        # Append first convolutional layer
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride_first_conv, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        # Append second convolutional layer
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        
        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x 
        out = self.layers(x)

        if self.stride_first_conv!=1:
            # Project the identity to match the correct dimension
            identity = self.bn(self.conv1x1(identity))

        return self.relu(out + identity)

class SkipBlock(nn.Module):
    '''
    Implements one block of the residual CNN described in the paper "Deep residual learning for image recognition" by He et al. (2015)
    Here block refers to the different coloured parts of the network in the figure of the paper.
    '''
    def __init__(self, in_ch, out_ch, kernel_size, num_layers, first_layer_stride=2):
        super(SkipBlock, self).__init__()
        layers = []

        # Append the skip connections
        for i in range(num_layers):
            if i == 0:
                layers.append(SkipLayer(in_ch, out_ch, kernel_size, stride_first_conv=first_layer_stride))
            else:
                layers.append(SkipLayer(out_ch, out_ch, kernel_size, stride_first_conv=1))
        
        self.layers = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.layers(x)

class SkipCNN(nn.Module):
    def __init__(self, config, n_classes=1):
        super(SkipCNN, self).__init__()
        self.config = config

        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = PlainBlock(3, 64, 7, config[0], first_layer_stride=2, padding=3)
        self.block2 = SkipBlock(64, 64, 3, config[1], first_layer_stride=1)
        self.block3 = SkipBlock(64, 128, 3, config[2], first_layer_stride=2)
        self.block4 = SkipBlock(128, 256, 3, config[3], first_layer_stride=2)
        self.block5 = SkipBlock(256, 512, 3, config[4], first_layer_stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    
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
        x = self.block1(x)
        print("First block")
        print(x.shape)
        x = self.pool(x)
        print("Pool")
        print(x.shape)
        x = self.block2(x)
        print("Second block")
        print(x.shape)
        x = self.block3(x)
        print("Third block")
        print(x.shape)
        x = self.block4(x)
        print("Fourth block")
        print(x.shape)
        x = self.block5(x)
        print("Fifth block")
        print(x.shape)
        x = self.avg_pool(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        
    def _getname_(self):
        return "SkipCNN_"+str(sum(self.config)*2)+"_layers"