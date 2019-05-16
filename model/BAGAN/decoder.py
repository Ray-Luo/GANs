import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, latent_size, img_dim, z_dim, init_resolution = 64):
        super(Decoder, self).__init__()

        self.init_resolution = init_resolution
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(in_features = latent_size,\
                             out_features = 1024,\
                             bias = False)
        self.relu4 = nn.LeakyReLU(negative_slope = 0.2)


        self.fc2 = nn.Linear(1024, init_resolution*init_resolution*z_dim, bias = False)
        self.relu5 = nn.LeakyReLU(0.2)

        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4 = nn.Conv2d(in_channels = z_dim, \
                               out_channels = z_dim*2,\
                               kernel_size = 3,\
                               stride = 1,\
                               padding = 1,\
                               bias = False)
        self.relu6 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(in_channels = z_dim*2, \
                               out_channels = z_dim,\
                               kernel_size = 3,\
                               stride = 1,\
                               padding = 1,\
                               bias = False)
        self.relu7 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(in_channels = z_dim, \
                               out_channels = 3,\
                               kernel_size = 3,\
                               stride = 1,\
                               padding = 1,\
                               bias = False)
        self.tanh0 = nn.Tanh()



    def forward(self, x): # x will be latent vector
        output0 = self.relu4(self.fc1(x))
        output1 = self.relu5(self.fc2(output0))
        output1 = output1.view(-1, self.z_dim, self.init_resolution, self.init_resolution)
        # print("output1 shape", output1.size())
        init_resolution = self.init_resolution
        while init_resolution != self.img_dim:

            if init_resolution < self.img_dim/2:
                # print("branch 1 in, output1 shape", output1.size())
                output1 = self.relu6(self.conv4(output1))
                # print("branch 1 out, output1 shape", output1.size())
            else:
                # print("branch 2 in, output1 shape", output1.size())
                output1 = self.relu7(self.conv5(output1))
                # print("branch 2 out, output1 shape", output1.size())

            output1 = self.upsample0(output1)
            init_resolution *= 2
            # print("out output1 shape", output1.size())

        output = self.tanh0(self.conv6(output1))
        # print("output shape", output.size())

        # this is the fake image
        return output 
                






