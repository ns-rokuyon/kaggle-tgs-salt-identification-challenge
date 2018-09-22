from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data import get_image


class ClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transformer = None

    def set_transformer(self, size=None):
        if size:
            self.transformer = transforms.Compose([
                transforms.Resize(size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        filename = self.df.loc[i, 'ImageId']
        encoded_pixels = self.df.loc[i, 'EncodedPixels']

        # NaN type is float
        label = 0 if type(encoded_pixels) == float else 1

        im = get_image(filename)
        im = self.transformer(im)

        return im, label