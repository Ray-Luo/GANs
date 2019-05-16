from BVDataset import BVDataset
from torch import nn, optim
import torch
import os
import sys
import numpy as np
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributions.multivariate_normal as mn
from model.BAGAN.encoder import Encoder
from model.BAGAN.decoder import Decoder


def reparameterization(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    z = eps.mul(std).add_(mu)
    return z

def batch2one(Z, y, z, class_num):
    for i in range(y.shape[0]):
        Z[y[i]] = torch.cat((Z[y[i]], z[i].cpu()), dim=0) # Z[label][0] should be deleted..
    return Z


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
seed = 1
log_interval = 50
model_save_path = "/home/kdd/Documents/GAN/model/BAGAN/ave_checkpoints/"
z_dim = 16
class_num = 95


train_root = '/home/kdd/Downloads/fruits/fruits-360/Training'
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
    batch_size = 1200, shuffle=True, **kwargs)



encoder = Encoder()
decoder = Decoder()

if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

# model = model.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)


for file in os.listdir(model_save_path):
    
    if file.startswith("encoder") and file.endswith(".tar"):
        checkpointE = torch.load(model_save_path + file, map_location='cpu')
        encoder.module.load_state_dict(checkpointE['encoder_state_dict'])
        encoder.to(device)

    if file.startswith("decoder") and file.endswith(".tar"):
        checkpointD = torch.load(model_save_path + file, map_location='cpu')
        decoder.module.load_state_dict(checkpointD['decoder_state_dict'])
        decoder.to(device)

encoder.eval()
decoder.eval()
Z = list()

with torch.no_grad():
    for i in range(class_num):
        Z.append(torch.zeros((1, z_dim), dtype=torch.float))

    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        mu, log_sigmoid = encoder(img)
        z = reparameterization(mu, log_sigmoid)
        z = z.view(-1, 1, z_dim)
        Z = batch2one(Z, label, z, class_num)

    N = []
    for i in range(class_num):
        label_mean = torch.mean(Z[i][1:], dim=0).double()
        label_cov = torch.from_numpy(np.cov(Z[i][1:].numpy(), rowvar=False)).double()
        m = mn.MultivariateNormal(label_mean, label_cov)
        sample = m.sample((2048, ))
        sample = sample.float()
        sample = sample.to(device).view(z_dim, -1)
        fake = decoder(sample)
        N.append(m)

    torch.save({
            'distribution': N
        }, model_save_path + "class_distribution.dt")