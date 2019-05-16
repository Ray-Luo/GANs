from BVDataset import BVDataset
from torch import nn, optim
import torch
import os
import sys
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
from model.BAGAN.encoder import Encoder
from model.BAGAN.decoder import Decoder

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
seed = 1
log_interval = 10
in_channels = 3
z_dim = 16
img_dim = 256
latent_size = 100
iters = 100000
model_save_path = "/home/kdd/Documents/GAN/model/BAGAN/ave_checkpoints/"


# train_img_folder = '/home/kdd/Documents/GAN/data/final/'
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

# # define training data for training autoencoder

# # let's train the autoencoder
# train_data = BVDataset(train_ids, train_labels, img_dim)
# train_loader_food = torch.utils.data.DataLoader(train_data, batch_size=150, shuffle=True, num_workers=2)




train_root = '/home/kdd/Documents/GAN/data/torch_dataloader/train'

data_transforms = transforms.Compose([
    # tv.transforms.RandomCrop((64, 64), padding=4),
    transforms.Resize((img_dim,img_dim)),
    transforms.ToTensor(),
])

train_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=data_transforms),
    batch_size = 150, shuffle=True, **kwargs)


encoder = Encoder(in_channels, z_dim, latent_size, img_dim)
decoder = Decoder(latent_size, img_dim, z_dim)

if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

encoder = encoder.to(device)
decoder = decoder.to(device)

optimizerE = optim.Adam(encoder.parameters(), lr=1e-3)
optimizerD = optim.Adam(decoder.parameters(), lr=1e-3)
loss_mse = nn.MSELoss(reduction="sum")

train_losses = []
min_loss = float("inf")
epoch = 0 

for file in os.listdir(model_save_path):

    if file.startswith("encoder") and file.endswith(".tar"):
        checkpointE = torch.load(model_save_path + file, map_location='cpu')
        encoder.module.load_state_dict(checkpointE['encoder_state_dict'])
        encoder.to(device)
        optimizerE.load_state_dict(checkpointE['encoder_optimizer'])
        min_loss = checkpointE['prev_loss']
        train_losses = checkpointE['train_losses']
        epoch = checkpointE['epoch']

    if file.startswith("decoder") and file.endswith(".tar"):
        checkpointD = torch.load(model_save_path + file, map_location='cpu')
        decoder.module.load_state_dict(checkpointD['decoder_state_dict'])
        decoder.to(device)
        optimizerD.load_state_dict(checkpointD['decoder_optimizer'])



def train(epoch, prev_loss):
    min_loss = prev_loss

    encoder.train()
    decoder.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_food):
        data = data.to(device)
        optimizerD.zero_grad()
        optimizerE.zero_grad()

        latent_feature = encoder(data)
        # print("latent_feature", latent_feature.size())
        fake = decoder(latent_feature)

        loss = loss_mse(fake, data)
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
    if loss < min_loss:
        min_loss = loss

        for file in os.listdir(model_save_path):
            if file.startswith("decoder") or file.startswith("encoder"):
                os.remove(model_save_path + file)

        torch.save({
            'epoch':           epoch,
            'encoder_state_dict':  encoder.module.state_dict(),
            'encoder_optimizer':   optimizerE.state_dict(),
            'train_losses':    train_losses,
            'prev_loss':       min_loss
            }, model_save_path + "encoder_" + '{:.4f}'.format(min_loss) + ".tar")


        torch.save({
            'decoder_state_dict':  decoder.module.state_dict(),
            'decoder_optimizer':   optimizerD.state_dict(),
            }, model_save_path + "decoder_" + '{:.4f}'.format(min_loss) + ".tar")


while epoch < iters:

    # train(epoch, min_loss)
    # min_loss = prev_loss

    encoder.train()
    decoder.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_food):
        data = data.to(device)
        optimizerD.zero_grad()
        optimizerE.zero_grad()

        latent_feature = encoder(data)
        # print("latent_feature", latent_feature.size())
        fake = decoder(latent_feature)

        loss = loss_mse(fake, data)
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
    if loss < min_loss:
        min_loss = loss

        for file in os.listdir(model_save_path):
            if file.startswith("decoder") or file.startswith("encoder"):
                os.remove(model_save_path + file)

        torch.save({
            'epoch':           epoch,
            'encoder_state_dict':  encoder.module.state_dict(),
            'encoder_optimizer':   optimizerE.state_dict(),
            'train_losses':    train_losses,
            'prev_loss':       min_loss
            }, model_save_path + "encoder_" + '{:.4f}'.format(min_loss) + ".tar")


        torch.save({
            'decoder_state_dict':  decoder.module.state_dict(),
            'decoder_optimizer':   optimizerD.state_dict(),
            }, model_save_path + "decoder_" + '{:.4f}'.format(min_loss) + ".tar")



    # test(epoch)
    with torch.no_grad():

        for file in os.listdir("/home/kdd/Documents/GAN/model/BAGAN/ave_results/"):
            if file.endswith(".png"):
                os.remove("/home/kdd/Documents/GAN/model/BAGAN/ave_results/" + file)


        sample = torch.randn(32, latent_size).to(device)

        # mu, logvar = encoder(sample)
        # z = reparameterization(mu, logvar)
        recon_batch = decoder(sample).cpu()

        # sample = model.module.decode(sample).cpu()
        save_image(recon_batch.view(32, 3, img_dim, img_dim),
                   '/home/kdd/Documents/GAN/model/BAGAN/ave_results/sample_' + str(epoch) + '.png')

    epoch += 1