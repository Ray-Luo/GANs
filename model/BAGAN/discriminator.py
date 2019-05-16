'''
Discrimator is very much like the encoder
'''

from torch import nn, optim

class Discriminator(nn.Module):

    def __init__(self, in_channels, z_dim, img_dim, n_class):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.conv0 = nn.Conv2d(in_channels = in_channels, \
                               out_channels = z_dim,\
                               kernel_size = 3,\
                               stride = 2,\
                               padding = 1,\
                               bias = True)
        self.bn0 = nn.BatchNorm2d(z_dim)
        self.relu0 = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout0 = nn.Dropout(p = 0.3)

        self.conv1 = nn.Conv2d(z_dim, z_dim*2, 3, 1, 1, True)
        self.bn1 = nn.BatchNorm2d(z_dim*2)
        self.relu1 = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout1 = nn.Dropout(p = 0.3)

        self.conv2 = nn.Conv2d(z_dim*2, z_dim*4, 3, 2, 1, True)
        self.bn2 = nn.BatchNorm2d(z_dim*4)
        self.relu2 = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout2 = nn.Dropout(p = 0.3)

        self.conv3 = nn.Conv2d(z_dim*4, z_dim*8, 3, 1, 1, True)
        self.bn3 = nn.BatchNorm2d(z_dim*8)
        self.relu3 = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout3 = nn.Dropout(p = 0.3)

        self.fc0 = nn.Linear(z_dim*8*64*64, n_class + 1) # n_class + 1. The last class is: FAKE
        self.softmax0 = nn.LogSoftmax()


    def forward(self, x):
        output0 = self.dropout0(self.relu0(self.bn0(self.conv0(x))))
        output1 = self.dropout1(self.relu1(self.bn1(self.conv1(output0))))
        output2 = self.dropout2(self.relu2(self.bn2(self.conv2(output1))))
        output3 = self.dropout3(self.relu3(self.bn3(self.conv3(output2))))
        output3 = output3.view(-1, self.z_dim*8*64*64)
        # print(output3.size())
        output = self.softmax0(self.fc0(output3.view(-1, self.z_dim*8*64*64)))
        
        # this is latent feature
        return output