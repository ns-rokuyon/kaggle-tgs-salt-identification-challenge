import torch
import gc
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from PIL import Image

data_root_dir = Path('D:/Users/ns/.kaggle/competitions/tgs-salt-identification-challenge')
image_dir = Path(data_root_dir / 'train')
train_csv = data_root_dir / 'train.csv'
test_csv = data_root_dir / 'sample_submission.csv'
depth_csv = data_root_dir / 'depths.csv'

model_dir = Path('D:/Users/ns/git_repos/kaggle-tgs-salt/models')


def get_dfs(seed=1):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    depth_df = pd.read_csv(depth_csv)

    # Append depth data
    train_df = pd.merge(train_df, depth_df, on='id')
    test_df = pd.merge(test_df, depth_df, on='id')

    # Append coverage data
    n_pixels = 101 * 101
    def _cov(_id):
        return (np.array(get_train_mask(_id), dtype=np.uint8) / 255).sum() / n_pixels
    train_df['coverage'] = train_df['id'].apply(_cov)

    def _cov2class(cov):
        for i in range(0, 11):
            if cov * 10 <= i:
                return i
    train_df['coverage_class'] = train_df['coverage'].apply(_cov2class)

    # Split train set to train and validation
    train_df, val_df = train_test_split(train_df, stratify=train_df['coverage_class'], test_size=0.2, random_state=seed)

    return train_df, val_df, test_df


def evaluate(model, loader, device):
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    predictions = np.empty((0, 1, 101, 101), dtype=np.int8)
    targets = np.empty((0, 1, 101, 101), dtype=np.int8)
    for data, target in loader:
        pred = predict(model, data, device=device)
        target = target.cpu().detach().numpy().astype(np.int8)

        predictions = np.concatenate([predictions, pred])
        targets = np.concatenate([targets, target])

    return jaccard_similarity_score(targets.flatten(), predictions.flatten())


def predict(model, x, device=None):
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((101, 101))

    x = x.to(device)
    logits = model(x)
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).cpu()
    pred = np.array([np.array(resize(to_pil(_))) for _ in pred])
    pred = pred.reshape(pred.shape[0], 1, 101, 101).astype(np.float32)
    return torch.Tensor(pred)


def save_model(model, keyname):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))
    
    filename = f'{keyname}.model'
    torch.save(model, str(model_dir / filename))


def get_train_image(_id):
    path = data_root_dir / 'train' / 'images' / f'{_id}.png'
    im = Image.open(str(path))
    return im


def get_train_mask(_id):
    path = data_root_dir / 'train' / 'masks' / f'{_id}.png'
    im = Image.open(str(path))
    return im


def get_test_image(_id):
    path = data_root_dir / 'test' / 'images' / f'{_id}.png'
    im = Image.open(str(path))
    return im