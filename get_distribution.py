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
in_channels = 3
z_dim = 16
img_dim = 256
latent_size = 100
log_interval = 50
model_save_path = "/home/kdd/Documents/GAN/model/BAGAN/ave_checkpoints/"
z_dim = 16
class_num = 2



train_root = '/home/kdd/Documents/GAN/data/torch_dataloader/train'

data_transforms = transforms.Compose([
    # tv.transforms.RandomCrop((64, 64), padding=4),
    transforms.Resize((img_dim,img_dim)),
    transforms.ToTensor(),
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=data_transforms),
    batch_size = 150, shuffle=True, **kwargs)



encoder = Encoder(in_channels, z_dim, latent_size, img_dim)
decoder = Decoder(latent_size, img_dim, z_dim)

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
covariances = list()
means = list()

n = 0

with torch.no_grad():
    for i in range(class_num):
        Z.append(torch.zeros((1, latent_size), dtype=torch.float))

    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        latent_feature = encoder(img)
        n += 1
        # print("1 ",latent_feature.size())
        # print(n)
        latent_feature = latent_feature.view(-1, 1, latent_size)
        # print("2 ",latent_feature.size())
        Z = batch2one(Z, label, latent_feature, class_num)

    for i in range(class_num):
        label_mean = torch.mean(Z[i][1:], dim=0).double()
        label_cov = torch.from_numpy(np.cov(Z[i][1:].numpy(), rowvar=False)).double()
        means.append(label_mean)
        covariances.append(label_cov)

    # print(len(covariances),covariances[0].size())
    torch.save({
            'means': means,
            'covariances': covariances
        }, model_save_path + "class_distribution.dt")