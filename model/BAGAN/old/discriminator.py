import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, z_dim, c_dim, df_dim, class_num):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(16*25*25, 95 + 1)
        self.softmax = nn.LogSoftmax()

        self.relu = nn.ReLU()


    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        # print(x.shape, conv1.shape, conv2.shape, conv3.shape, conv4.shape)
        # conv4 = self.relu(conv4).view(-1, 16*25*25) #16*64*64

        # fc1 = self.fc1(conv4)
        # fc1 = self.fc_bn1(fc1)
        # fc1 = self.relu(fc1)

        # r1 = self.fc21(fc1)
        # r2 = self.fc22(fc1)
        output = self.softmax(self.fc1(conv4.view(-1, 16*25*25)))
        
        return conv4, output
