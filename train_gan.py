import argparse
from torch import nn, optim
import torch
import os
import sys
import math
import numpy as np
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
    parser.add_argument('--train_ae',    type = bool,                                                  help = "if train the autoencoder")
    parser.add_argument('--ae_iter',     type = int,  default = 10000,                                 help = "number of iterations for training the autoencoder")
    parser.add_argument('--z_dim',       type = int,  default = 16,                                    help = "latent vector dimension")
    parser.add_argument('--c_dim',       type = int,  default = 3,                                     help = "number of channels")
    parser.add_argument('--gf_dim',      type = int,  default = 16,                                    help = "generator input dimension")
    parser.add_argument('--df_dim',      type = int,  default = 16,                                    help = "discriminator input dimension")
    parser.add_argument('--img_size',    type = int,  default = 256,                                   help = "input img dimension")
    parser.add_argument('--latent_size', type = int,  default = 100,                                   help = "lantent_size")
    parser.add_argument('--class_num',   type = int,  default = 2,                                    help = "number of class")
    parser.add_argument('--gan_iter',    type = int,  default = 10000,                                 help = "number of iterations for training the GAN")
    parser.add_argument('--use_gpu',     type = bool, default = True,                                  help = "if use GPU")
    parser.add_argument('--print_every', type = int,  default = 10,                                    help = "print loss every k step")
    parser.add_argument('--batch_size',  type = int,  default = 1200,                                  help = "")
    args = parser.parse_args()


    train_root = '/home/kdd/Documents/GAN/data/torch_dataloader/train'
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    data_transforms = transforms.Compose([
        # tv.transforms.RandomCrop((64, 64), padding=4),
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_root, transform=data_transforms),
        batch_size = 100, shuffle=True, **kwargs)


    G = Generator(args.latent_size, args.img_size, args.z_dim)
    D = Discriminator(in_channels=args.c_dim, z_dim=args.z_dim, img_dim=args.img_size, n_class=args.class_num)  #in_channels, z_dim, img_dim, n_class

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

    D.train()
    G.train()

    D_loss_list = []
    G_loss_list = []

    real_label = torch.LongTensor(args.batch_size)
    fake_label = torch.LongTensor(args.batch_size)

    prev_loss = float("inf")
    epoch = 0

    for file in os.listdir(args.gan_dir):

        if file.startswith("gan_G") and file.endswith(".tar"):
            checkpointG = torch.load(args.gan_dir + file, map_location='cpu')
            G.module.load_state_dict(checkpointG['G_state_dict'])
            G.to(device)
            optimizerG.load_state_dict(checkpointG['G_optimizer'])
            prev_loss = checkpointG['prev_loss']
            G_loss_list = checkpointG['G_losses']
            epoch = checkpointG['epoch']

        if file.startswith("gan_D") and file.endswith(".tar"):
            checkpointD = torch.load(args.gan_dir + file, map_location='cpu')
            D.module.load_state_dict(checkpointD['D_state_dict'])
            D.to(device)
            optimizerD.load_state_dict(checkpointD['D_optimizer'])
            D_loss_list = checkpointD['D_losses']



    while epoch < args.gan_iter:

        epoch_disc_loss = []
        epoch_gen_loss = []

        for batch_idx, (img, label) in enumerate(train_loader):
            batch_size = img.size()[0]
            img = img.to(device)
            label = label.to(device)

            ################## Train Discriminator ##################
            fake_size = int(np.ceil(batch_size*1.0/args.class_num))

            # sample some labels from p_c, then latent and images
            sample_labels = torch.randint(0, args.class_num-1, (fake_size,), dtype=torch.long).to(device)
            latent_gen = generate_latent(sample_labels, means, covariances)
            generated_images = G(latent_gen)

            sample_labels = torch.from_numpy(np.full(sample_labels.size()[0], args.class_num)).type(torch.LongTensor).to(device)
            new_X = torch.cat((img, generated_images), dim=0)
            new_label = torch.cat((label, sample_labels), dim=0)
            predicted_label = D(new_X)
            d_loss = criterion(predicted_label, new_label)
            epoch_disc_loss.append(d_loss.item())

            D.zero_grad()
            G.zero_grad()
            d_loss.backward()
            optimizerD.step()

            ################## Train Generator ##################
            sample_labels = torch.randint(0, args.class_num-1, (fake_size+batch_size,), dtype=torch.long).to(device)
            latent_gen = generate_latent(sample_labels, means, covariances)
            generated_images = G(latent_gen)
            predicted_label = D(generated_images)
            g_loss = criterion(predicted_label, sample_labels)
            epoch_gen_loss.append(g_loss.item())

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            optimizerG.step()


            if batch_idx % args.print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\td_loss: {:.6f}\tg_loss: {:.6f}'.format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           d_loss.item(), g_loss.item()))

        train_disc_loss = np.mean(np.array(epoch_disc_loss))
        train_gen_loss = np.mean(np.array(epoch_gen_loss))

        D_loss_list.append(train_disc_loss)
        G_loss_list.append(train_gen_loss)
        print('====> Epoch: {} Average d_loss: {:.4f} Average g_loss: {:.4f}'.format(epoch, train_disc_loss, train_gen_loss))

        # if loss decreases, we save the model
        loss = (train_disc_loss + train_gen_loss)/2
        if loss < prev_loss:
            prev_loss = loss

            for file in os.listdir(args.gan_dir):
                if file.startswith("gan"):
                    os.remove(args.gan_dir + file)

            torch.save({
                'epoch':         epoch,
                'G_state_dict':  G.module.state_dict(),
                'G_optimizer':   optimizerG.state_dict(),
                'G_losses':      G_loss_list,
                'prev_loss':     prev_loss
                }, args.gan_dir + "gan_G_" + '{:.4f}'.format(prev_loss) + ".tar")


            torch.save({
                'D_state_dict':  D.module.state_dict(),
                'D_optimizer':   optimizerD.state_dict(),
                'D_losses':      D_loss_list,
                }, args.gan_dir + "gan_D_" + '{:.4f}'.format(prev_loss) + ".tar")


        epoch += 1

