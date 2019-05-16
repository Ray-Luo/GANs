import torch
from torch.utils import data
from PIL import Image
from PIL import ImageOps
import numpy as np
# import torchvision.transforms.Resize as Resize

class BVDataset(data.Dataset):

    def __init__(self, ids, labels, expected_img_size):
        self.ids = ids
        self.labels = labels
        self.expected_img_size = expected_img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        img = Image.open(img_path)
        img = self.padding(img)
        size = self.expected_img_size, self.expected_img_size
        img.thumbnail(size, Image.BICUBIC)
        img = np.array(img)
        # img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        label = np.array(self.labels[index])
        label = torch.from_numpy(label).float()

        return (img, label)

    def padding(self, img):
        desired_size = self.expected_img_size
        delta_width = desired_size - img.size[0]
        delta_height = desired_size - img.size[1]
        pad_width = delta_width //2
        pad_height = delta_height //2
        padding = (pad_width,pad_height,delta_width-pad_width,delta_height-pad_height)
        return ImageOps.expand(img, padding)
