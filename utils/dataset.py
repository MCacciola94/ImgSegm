
from torch.utils.data import DataLoader, Dataset
from .utils import read_image_mask
import torch
import numpy as np
import torchvision.transforms as transforms

#Dataset class
class HubMapDataset(Dataset):
    def __init__(self, df, transform= {"img": transforms.ToTensor(), "mask": None}):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        id_ = self.df["id"][idx]
        image , mask = read_image_mask(self.df, id_)
        image = image.astype(np.single)
        mask = mask.astype(np.single)
        
        if self.transform["img"] is not None:
            image= self.transform["img"](image)
        if self.transform["mask"] is not None:
            mask= self.transform["mask"](mask)

        return image, mask


