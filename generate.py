import argparse
from torch import nn, optim
import torch
import os
import sys
import math
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
from model.BAGAN.encoder import Encoder
from model.BAGAN.decoder import Decoder
from model.BAGAN.generator import Generator
from model.BAGAN.discriminator import Discriminator
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',   type = str,  default = "/home/kdd/Documents/GAN/data/train/", help = "trian dataset location")
    parser.add_argument('--test_dir',    type = str,  default = "/home/kdd/Documents/GAN/data/test/",  help = "test dataset location")
    parser.add_argument('--save_dir',    type = str,  default = "/home/kdd/Documents/GAN/data/train/", help = "model save location")
    parser.add_argument('--ae_dir',      type = str,  default = "/home/kdd/Documents/GAN/model/BAGAN/ave_checkpoints/", help = "autoencoder location")
    parser.add_argument('--gan_dir',     type = str,  default = "/home/kdd/Documents/GAN/model/BAGAN/gan_checkpoints/", help = "GAN location")
    parser.add_argument('--save_img_dir',type = str,  default = "/home/kdd/Documents/GAN/model/BAGAN/produced_img/", help = "GAN location")
    parser.add_argument('--z_dim',       type = int,  default = 16,                                    help = "latent vector dimension")
    parser.add_argument('--c_dim',       type = int,  default = 3,                                     help = "number of channels")
    parser.add_argument('--gf_dim',      type = int,  default = 16,                                    help = "generator input dimension")
    parser.add_argument('--df_dim',      type = int,  default = 16,                                    help = "discriminator input dimension")
    parser.add_argument('--img_size',    type = int,  default = 256,                                   help = "input img dimension")
    parser.add_argument('--latent_size', type = int,  default = 100,                                   help = "lantent_size")
    parser.add_argument('--class_num',   type = int,  default = 2,                                    help = "number of class")
    parser.add_argument('--use_gpu',     type = bool, default = True,                                  help = "if use GPU")
    parser.add_argument('--sample_num',  type = int,  default = 10,                                    help = "number of minor img to produce")
    parser.add_argument('--minor_cls',   type = int,  default = 1,                                    help = "minority class to produce")
    args = parser.parse_args()


    cuda = torch.cuda.is_available()

    G = Generator(args.latent_size, args.img_size, args.z_dim)
    D = Discriminator(in_channels=args.c_dim, z_dim=args.z_dim, img_dim=args.img_size, n_class=args.class_num)

    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    if args.use_gpu and cuda:
        device = torch.device("cuda" if cuda else "cpu")
        G = G.to(device)
        D = D.to(device)

    # load weights from autoencoder
    checkpointD = torch.load(args.ae_dir + "decoder_229.2825.tar", map_location='cpu')['decoder_state_dict']
    # for k, v in checkpointD.items():
    #     print(k, v.size())
    # print(G)
    G.module.load_state_dict(checkpointD)

    checkpointE = torch.load(args.ae_dir + "encoder_229.2825.tar", map_location='cpu')['encoder_state_dict']
    del checkpointE['fc0.weight']
    del checkpointE['fc0.bias']
    checkpointE.update({'fc0.weight':D.module.state_dict()['fc0.weight']})
    checkpointE.update({'fc0.bias':D.module.state_dict()['fc0.bias']})
    D.module.load_state_dict(checkpointE)

    means = torch.load(args.ae_dir + "class_distribution.dt", map_location='cpu')['means']
    covariances = torch.load(args.ae_dir + "class_distribution.dt", map_location='cpu')['covariances']

    criterion = nn.NLLLoss()

    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    D.eval()
    G.eval()


    for file in os.listdir(args.gan_dir):

        if file.startswith("gan_G") and file.endswith(".tar"):
            checkpointG = torch.load(args.gan_dir + file, map_location='cpu')
            G.module.load_state_dict(checkpointG['G_state_dict'])
            G.to(device)

        if file.startswith("gan_D") and file.endswith(".tar"):
            checkpointD = torch.load(args.gan_dir + file, map_location='cpu')
            D.module.load_state_dict(checkpointD['D_state_dict'])
            D.to(device)


    for i in range(args.sample_num):
        sample_labels = torch.LongTensor([args.minor_cls]).to(device)
        latent_gen = generate_latent(sample_labels, means, covariances)
        generated_image = G(latent_gen).squeeze(0).permute(1,2,0).cpu().data.numpy()
        generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
        generated_image = Image.fromarray(generated_image)
        generated_image.save(args.save_img_dir + str(args.minor_cls) + "_" + str(i) + ".png", 'PNG')
