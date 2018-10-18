import numpy as np
import albumentations as alb
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import joint_transform as jt
from data import *


def mask_to_label(mask, n_pixels):
    ratio = mask.numpy().sum() / n_pixels
    if 0 < ratio < 0.05:
        label = 1
    elif 0.05 <= ratio:
        label = 2
    else:
        label = 0
    return label


class SegmentationDataset(Dataset):
    def __init__(self, df, size=(128, 128),
                 use_depth_channels=False,
                 with_aux_label=False,
                 as_aux_label='cov',
                 use_augmentation=False):
        self.df = df
        self.size = size
        self.use_depth_channels = use_depth_channels
        self.with_aux_label = with_aux_label
        self.as_aux_label = as_aux_label    # cov: mask_to_label, coverage_class: get_mask_type() value
        self.use_augmentation = use_augmentation
        self.transformer = jt.Compose([
            jt.Grayscale(),
            jt.FreeScale(self.size),
            jt.RandomHorizontallyFlip(),
            jt.ToTensor()
        ])
        if self.use_augmentation:
            print('Use augmentations')
            self.aug = alb.Compose([
                #alb.RandomSizedCrop(min_max_height=(80, 101), height=101, width=101, p=0.2),
                #alb.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                #alb.GridDistortion(p=0.5),
                #alb.Blur(),
                #alb.GaussNoise(),
                alb.RandomBrightness(),
                alb.RandomGamma()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _id = self.df.iloc[i]['id']

        im = get_train_image(_id)
        mask = get_train_mask(_id)

        if self.use_augmentation:
            augmented = self.aug(image=np.array(im),
                                 mask=np.array(mask))
            im = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        im, mask = self.transformer(im, mask)

        if self.use_depth_channels:
            im = add_depth_channels(im)

        if self.with_aux_label:
            if self.as_aux_label == 'cov':
                label = mask_to_label(mask, self.size[0] * self.size[1])
            elif self.as_aux_label == 'coverage_class':
                label = self.df.iloc[i]['coverage_class']
            else:
                raise ValueError(self.as_aux_label)

            return im, mask, label

        return im, mask


class SegmentationInferenceDataset(Dataset):
    """
    Input tensor: resize to specified size
    GT tensor: no resize
    """
    def __init__(self, df, input_size=(128, 128),
                 use_depth_channels=False,
                 with_aux_label=False,
                 with_gt=True,
                 with_raw_input=False):
        self.df = df
        self.input_size = input_size
        self.use_depth_channels = use_depth_channels
        self.with_aux_label = with_aux_label
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
            if self.with_aux_label:
                label = mask_to_label(mask, self.input_size[0] * self.input_size[1])
                return self.raw_input_transformer(_im), im, mask, label
            return self.raw_input_transformer(_im), im, mask

        if self.with_aux_label:
            label = mask_to_label(mask, self.input_size[0] * self.input_size[1])
            return im, mask, label
        return im, mask