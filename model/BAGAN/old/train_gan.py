import argparse
from torch import nn, optim
import torch
import os
import sys
import math
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
from model.BAGAN.encoder import Encoder
from model.BAGAN.decoder import Decoder
from model.BAGAN.generator import Generator
from model.BAGAN.discriminator import Discriminator



def conditional_latent_generator(distribution, class_num, batch):
    class_labels = torch.randint(0, class_num, (batch,), dtype=torch.long)
    print(len(distribution))
    # print(class_labels[0].item())
    # fake_z = distribution[class_labels[0].item()].sample((2048,))
    # for c in class_labels:
    #     fake_z = torch.cat((fake_z, distribution[c.item()].sample((2048,))), dim=0)
    return fake_z, class_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',   type = str,  default = "/home/kdd/Documents/GAN/data/train/", help = "trian dataset location")
    parser.add_argument('--test_dir',    type = str,  default = "/home/kdd/Documents/GAN/data/test/",  help = "test dataset location")
    parser.add_argument('--save_dir',    type = str,  default = "/home/kdd/Documents/GAN/data/train/", help = "model save location")
    parser.add_argument('--ae_dir',      type = str,  default = "/home/kdd/Documents/GAN/model/BAGAN/ave_checkpoints/", help = "autoencoder location")
    parser.add_argument('--train_ae',    type = bool,                                                  help = "if train the autoencoder")
    parser.add_argument('--ae_iter',     type = int,  default = 10000,                                 help = "number of iterations for training the autoencoder")
    parser.add_argument('--z_dim',       type = int,  default = 16,                                    help = "latent vector dimension")
    parser.add_argument('--c_dim',       type = int,  default = 3,                                     help = "number of channels")
    parser.add_argument('--gf_dim',      type = int,  default = 16,                                    help = "generator input dimension")
    parser.add_argument('--df_dim',      type = int,  default = 16,                                    help = "discriminator input dimension")
    parser.add_argument('--img_size',    type = int,  default = 512,                                   help = "input img dimension")
    parser.add_argument('--class_num',   type = int,  default = 95,                                    help = "number of class")
    parser.add_argument('--gan_iter',    type = int,  default = 10000,                                 help = "number of iterations for training the GAN")
    parser.add_argument('--use_gpu',     type = bool, default = True,                                  help = "if use GPU")
    parser.add_argument('--print_every', type = int,  default = 10,                                    help = "print loss every k step")
    parser.add_argument('--batch_size',  type = int,  default = 1200,                                  help = "")
    args = parser.parse_args()


    train_root = '/home/kdd/Downloads/fruits/fruits-360/Training'
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
        batch_size = args.batch_size, shuffle=True, **kwargs)

    G = Generator(args.z_dim, args.c_dim, args.gf_dim)
    D = Discriminator(args.z_dim, args.c_dim, args.df_dim, args.class_num)

    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    if args.use_gpu and cuda:
        device = torch.device("cuda" if cuda else "cpu")
        G = G.to(device)
        D = D.to(device)

    # load weights from autoencoder
    checkpointD = torch.load(args.ae_dir + "decoder_70.tar", map_location='cpu')['decoder_state_dict']
    G.module.load_state_dict(checkpointD)

    checkpointE = torch.load(args.ae_dir + "encoder_70.tar", map_location='cpu')['encoder_state_dict']
    del checkpointE['fc21.weight']
    del checkpointE['fc21.bias']
    del checkpointE['fc22.weight']
    del checkpointE['fc22.bias']
    del checkpointE['fc_bn1.weight']
    del checkpointE['fc_bn1.bias']
    del checkpointE['fc_bn1.running_var']
    del checkpointE['fc_bn1.running_mean']
    del checkpointE['fc_bn1.num_batches_tracked']
    checkpointE.update({'fc1.weight':D.module.state_dict()['fc1.weight']})
    checkpointE.update({'fc1.bias':D.module.state_dict()['fc1.bias']})

    D.module.load_state_dict(checkpointE)

    distribution = torch.load(args.ae_dir + "class_distribution.dt", map_location='cpu')['distribution']

    criterion = nn.NLLLoss()

    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    D.train()
    G.train()

    D_loss_list = []
    G_loss_list = []

    real_label = torch.LongTensor(args.batch_size)
    fake_label = torch.LongTensor(args.batch_size)

    epoch = 0
    while epoch < 10000:
        total_real = 0
        total_fake = 0
        correct_real = 0
        correct_fake = 0

        for i, (img, label) in enumerate(train_loader):
            batch_size = img.size(0)
            fake_num = math.ceil(args.batch_size/args.class_num)
            conditional_z, z_label = conditional_latent_generator(distribution, args.class_num, args.batch_size)
            label = label.long().squeeze()

            if args.use_gpu and cuda:
                img = img.to(device)
                label = label.to(device)

            sample_features, D_real = D(img)
            real_label.resize_(args.batch_size).copy_(label)

            if args.use_gpu and cuda:
                real_label = real_label.to(device)

            D_loss_real = criterion(D_real, real_label)
            noise = conditional_z[0:fake_num*2048].view(-1, args.z_dim, 2048)

            fake_label.resize_(noise.shape[0]).fill_(args.class_num)

            if args.use_gpu and cuda:
                noise = noise.to(device).float()
                fake_label = fake_label.to(device)
                z_label = z_label.to(device)

            fake = G(noise)

            _, D_fake = D(fake.detach())
            D_loss_fake = criterion(D_fake, fake_label)

            D_loss = D_loss_real + D_loss_fake
            D_losses.update(D_loss.item())
            D.zero_grad()
            G.zero_grad()
            D_loss.backward()
            optimizerD.step()

            noise = conditional_z.view(-1, args.z_dim, 1, 1)
            if args.use_gpu and cuda:
                noise = noise.to(device)

            fake = G(noise)
            _, D_fake = D(fake)
            G_loss = criterion(D_fake, z_label)
            G_losses.update(G_loss.data[0])

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            optimizerG.step()


            pred_real = torch.max(D_real.data, 1)[1]
            pred_fake = torch.max(D_fake.data, 1)[1]
            total_real += real_label.size(0)
            total_fake += z_label.size(0)
            correct_real += (pred_real == real_label).sum().item()
            correct_fake += (pred_fake == z_label).sum().item()


            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDLoss: {:.6f}\tGLoss: {:.6f}'.format(
                    epoch, i * len(img), len(train_loader.dataset),
                           100. * i / len(train_loader),
                           D_losses.item() / len(img), G_losses.item() / len(img)))


        # loss = train_loss / len(train_loader_food.dataset)
        # print('====> Epoch: {} Average Dloss: {:.4f}, Average Gloss: {:.4f}'.format(
        #     epoch, train_loss / len(train_loader_food.dataset)))
        # train_losses.append(loss)

        D_loss_list.append(D_losses.avg)
        G_loss_list.append(G_losses.avg)
        D_losses.reset()
        G_losses.reset()









            


        epoch += 1

