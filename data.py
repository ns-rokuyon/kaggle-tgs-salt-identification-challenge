import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

data_root_dir = Path('D:/Users/ns/.kaggle/competitions/tgs-salt-identification-challenge')
image_dir = Path(data_root_dir / 'train')
train_csv = data_root_dir / 'train.csv'
test_csv = data_root_dir / 'sample_submission.csv'
depth_csv = data_root_dir / 'depths.csv'

model_dir = Path('D:/Users/ns/git_repos/kaggle-tgs-salt/models')


def save_model(model, keyname):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))
    
    filename = f'{keyname}.model'
    torch.save(model, str(model_dir / filename))


def get_image(filename):
    path = data_root_dir / 'train' / filename
    im = Image.open(str(path))
    return im
