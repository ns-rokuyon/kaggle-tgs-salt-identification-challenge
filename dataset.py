from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import joint_transform as jt
from data import *


class SegmentationDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode
        self.transformer = None
        assert self.mode in ('train', 'val', 'test')

    def set_transformer(self, size=None):
        if self.mode == 'test':
            return self

        if self.mode == 'val':
            if size:
                self.transformer = jt.Compose([
                    jt.Grayscale(),
                    jt.FreeScale(size),
                    jt.ToTensor()
                ])
            else:
                self.transformer = jt.Compose([
                    jt.Grayscale(),
                    jt.ToTensor()
                ])

        if self.mode == 'train':
            if size:
                self.transformer = jt.Compose([
                    jt.Grayscale(),
                    jt.FreeScale(size),
                    jt.RandomHorizontallyFlip(),
                    jt.RandomRotateRightAngle(),
                    jt.ToTensor()
                ])
            else:
                self.transformer = jt.Compose([
                    jt.Grayscale(),
                    jt.RandomHorizontallyFlip(),
                    jt.RandomRotateRightAngle(),
                    jt.ToTensor()
                ])
        return self

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _id = self.df.iloc[i]['id']

        if self.mode == 'test':
            im = get_test_image(_id)
            return im

        im = get_train_image(_id)
        mask = get_train_mask(_id)

        im, mask = self.transformer(im, mask)

        return im, mask