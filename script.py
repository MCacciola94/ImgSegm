from paths_config import config_org as config
import numpy as np
import pandas as pd
import tifffile as tiff
from utils.dataset import HubMapDataset as hmd
from torch.utils.data import  DataLoader
import torch
import torchvision.transforms as transforms
from architectures import unet
import trainer
from PIL import Image

lr =0.1
batch_size = 16
save_dir = "./saves/script"
print_freq = 3
epochs = 60

data = pd.read_csv(config.TRAIN_CSV_PATH)
data_t = data[:280]
data_v =data[280:].reset_index()

trsfm = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor()
            ,transforms.Resize([256,256], interpolation=Image.NEAREST)
           # normalize,
        ])

trsfi = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor()
            ,transforms.Resize([256,256])
           # normalize,
        ])

dst =hmd(data_t, transform = {"img":trsfi,"mask":trsfm})

dsv =hmd(data_v, transform = {"img":trsfi,"mask":trsfm})

dl_t = DataLoader(dst, batch_size = batch_size)
dl_v = DataLoader(dsv, batch_size = 32)

en=enumerate(dl_t)
_,(img,mask) = next(en)
#breakpoint()
print("Net(x)")
net = unet.UNet(3,1)
net.cuda()
out = net(img.cuda())


# define loss function (criterion) and optimizer
criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], last_epoch= - 1)
#breakpoint()
tr = trainer.Trainer(net, {"train_loader":dl_t, "valid_loader": dl_v}, criterion, optimizer, lr_scheduler, save_dir, print_freq )
tr.train(epochs)
