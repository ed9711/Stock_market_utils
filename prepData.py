import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


class GetData(Dataset):

    def __init__(self, image_path, masks):
        self.masks = masks
        self.images_path = image_path
        # self.data_len = len(self.images_path) / 15
        self.data_len = len(self.images_path) - 5

    def __getitem__(self, index):
        # single_img = self.images_path.loc[index * 15:index * 15 + 14]
        single_img = self.images_path.loc[index:index+4]
        img_np = np.array(single_img)
        # img_np = [image / img_np.max() for image in img_np]
        img_tensors = torch.Tensor(img_np).float().cuda()
        mask_tensors = torch.Tensor([int(self.masks[index])]).long().cuda()
        return (img_tensors, mask_tensors)

    def __len__(self):
        return self.data_len


def prep_data(df, labels, percentage):
    dataset = GetData(df, labels)
    dataset_size = int(dataset.data_len)
    indices = list(range(dataset_size))
    split = int(np.floor(percentage * dataset_size))
    np.random.seed(420)
    np.random.shuffle(indices)
    train_i, val_i = indices[:dataset_size - split], indices[dataset_size - split:]
    # train_i, val_i = indices[:], indices[:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_i)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_i)
    trainLoad = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, batch_size=2, sampler=train_sampler,
                                            drop_last=True)
    validLoad = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, batch_size=2, sampler=valid_sampler,
                                            drop_last=True)

    return trainLoad, validLoad
