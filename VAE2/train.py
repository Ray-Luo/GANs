from BVDataset import BVDataset
from vae import VAE_CNN
from loss import customLoss
from torch import nn, optim
import torch
import os
import sys
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
from encoder import Encoder
from decoder import Decoder


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
seed = 1
log_interval = 50
model_save_path = "./checkpoints/"

# train_img_folder = "/home/kdd/Documents/GAN/data/train/"
# train_ids = list()
# train_labels = list()


# # prepare data
# for img in os.listdir(train_img_folder):
#     if img.endswith(".jpeg"):
#         train_ids.append(train_img_folder + img)
#         if "undamaged" in img:
#             train_labels.append(0)
#         else:
#             train_labels.append(1)

# define training data for training autoencoder

# # let's train the autoencoder
# train_data = BVDataset(train_ids, train_labels, 512)
# train_loader_food = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

train_root = '/home/kdd/Downloads/fruits/fruits-360/Training'
train_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
    batch_size = 1200, shuffle=True, **kwargs)


# model = VAE_CNN()
encoder = Encoder()
decoder = Decoder()

if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

# model = model.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizerE = optim.Adam(encoder.parameters(), lr=1e-3)
optimizerD = optim.Adam(decoder.parameters(), lr=1e-3)
loss_mse = customLoss()

train_losses = []
prev_loss = float("inf")
epoch = 0 

for file in os.listdir(model_save_path):
    if file.startswith("encoder") and file.endswith(".tar"):
        # checkpoint = torch.load(model_save_path + file, map_location='cpu')
        # model.module.load_state_dict(checkpoint['vae_state_dict'])
        # model.to(device)
        # optimizer.load_state_dict(checkpoint['vae_optimizer'])
        # prev_loss = checkpoint['prev_loss']
        # train_losses = checkpoint['train_losses']
        # epoch = checkpoint['epoch']

        checkpointE = torch.load(model_save_path + file, map_location='cpu')
        encoder.module.load_state_dict(checkpointE['encoder_state_dict'])
        encoder.to(device)
        optimizerE.load_state_dict(checkpointE['encoder_optimizer'])
        prev_loss = checkpointE['prev_loss']
        train_losses = checkpointE['train_losses']
        epoch = checkpointE['epoch']

    if file.startswith("decoder") and file.endswith(".tar"):
        # checkpoint = torch.load(model_save_path + file, map_location='cpu')
        # model.module.load_state_dict(checkpoint['vae_state_dict'])
        # model.to(device)
        # optimizer.load_state_dict(checkpoint['vae_optimizer'])
        # prev_loss = checkpoint['prev_loss']
        # train_losses = checkpoint['train_losses']
        # epoch = checkpoint['epoch']

        checkpointD = torch.load(model_save_path + file, map_location='cpu')
        decoder.module.load_state_dict(checkpointD['decoder_state_dict'])
        decoder.to(device)
        optimizerD.load_state_dict(checkpointD['decoder_optimizer'])


def reparameterization(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    z = eps.mul(std).add_(mu)
    return z


def train(epoch, prev_loss):
    # model.train()

    encoder.train()
    decoder.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_food):
        data = data.to(device)
        # optimizer.zero_grad()
        optimizerD.zero_grad()
        optimizerE.zero_grad()

        # recon_batch, mu, logvar = model(data)
        mu, logvar = encoder(data)
        z = reparameterization(mu, logvar)
        recon_batch = decoder(z)


        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # optimizer.step()

        optimizerE.step()
        optimizerD.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item() / len(data)))

    loss = train_loss / len(train_loader_food.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_food.dataset)))
    train_losses.append(loss)

    # if loss decreases, we save the model
    if loss < prev_loss:
        prev_loss = loss

        for file in os.listdir(model_save_path):
            if file.endswith(".tar"):
                os.remove(model_save_path + file)

        torch.save({
            'epoch':           epoch,
            'encoder_state_dict':  encoder.module.state_dict(),
            'encoder_optimizer':   optimizerE.state_dict(),
            'train_losses':    train_losses,
            'prev_loss':       prev_loss
            }, model_save_path + "encoder_" + str(int(loss)) + ".tar")

        torch.save({
            'decoder_state_dict':  decoder.module.state_dict(),
            'decoder_optimizer':   optimizerD.state_dict(),
            }, model_save_path + "decoder_" + str(int(loss)) + ".tar")



while epoch < 100000:

    train(epoch, prev_loss)
    # test(epoch)
    with torch.no_grad():

        for file in os.listdir("./results/"):
            if file.endswith(".png"):
                os.remove("./results/" + file)


        sample = torch.randn(32, 2048).to(device)

        # mu, logvar = encoder(sample)
        # z = reparameterization(mu, logvar)
        recon_batch = decoder(sample).cpu()

        # sample = model.module.decode(sample).cpu()
        save_image(recon_batch.view(32, 3, 100, 100),
                   './results/sample_' + str(epoch) + '.png')

    epoch += 1