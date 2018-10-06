from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import joint_transform as jt
from data import *


class SegmentationDataset(Dataset):
    def __init__(self, df, size=(128, 128), use_depth_channels=False):
        self.df = df
        self.size = size
        self.use_depth_channels = use_depth_channels
        self.transformer = jt.Compose([
            jt.Grayscale(),
            jt.FreeScale(self.size),
            jt.RandomHorizontallyFlip(),
            jt.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _id = self.df.iloc[i]['id']

        im = get_train_image(_id)
        mask = get_train_mask(_id)

        im, mask = self.transformer(im, mask)

        if self.use_depth_channels:
            im = add_depth_channels(im)

        return im, mask


class SegmentationInferenceDataset(Dataset):
    """
    Input tensor: resize to specified size
    GT tensor: no resize
    """
    def __init__(self, df, input_size=(128, 128), use_depth_channels=False, with_gt=True, with_raw_input=False):
        self.df = df
        self.input_size = input_size
        self.use_depth_channels = use_depth_channels
        self.with_gt = with_gt
        self.with_raw_input = with_raw_input
        self.input_transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.raw_input_transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.gt_transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _id = self.df.iloc[i]['id']

        if not self.with_gt:
            _im = get_test_image(_id)
            im = self.input_transformer(_im)
            if self.use_depth_channels:
                im = add_depth_channels(im)

            if self.with_raw_input:
                return self.raw_input_transformer(_im), im
            return im

        _im = get_train_image(_id)
        _mask = get_train_mask(_id)

        im = self.input_transformer(_im)
        mask = self.gt_transformer(_mask)

        if self.use_depth_channels:
            im = add_depth_channels(im)

        if self.with_raw_input:
            return self.raw_input_transformer(_im), im, mask
        return im, mask