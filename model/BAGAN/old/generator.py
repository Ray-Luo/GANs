import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim):
        super(Generator, self).__init__()

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        # self.relu0 = nn.ReLU(inplace=True)

        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        # self.relu2 = nn.ReLU(inplace=True)

        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(gf_dim)
        self.relu = nn.ReLU()

        # self.convTrans4 = nn.ConvTranspose2d(gf_dim, c_dim, 4, 2, 1, bias=False)
        # self.tanh = nn.Tanh()

        self.fc3 = nn.Linear(2048, 2048)
        self.fc_bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 16*25*25)
        self.fc_bn4 = nn.BatchNorm1d(16*25*25)


    def forward(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 25, 25)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        res = self.conv8(conv7).view(-1, 3, 100, 100)
        # print(z.shape)
        return res