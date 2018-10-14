import torch
import gc
import cv2
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import jaccard_similarity_score
from PIL import Image

data_root_dir = Path('D:/Users/ns/.kaggle/competitions/tgs-salt-identification-challenge')
image_dir = Path(data_root_dir / 'train')
train_csv = data_root_dir / 'train.csv'
test_csv = data_root_dir / 'sample_submission.csv'
depth_csv = data_root_dir / 'depths.csv'

model_dir = Path('D:/Users/ns/git_repos/kaggle-tgs-salt/models')
submission_dir = Path('D:/Users/ns/git_repos/kaggle-tgs-salt/submissions')
kfold_list_dir = Path('D:/Users/ns/git_repos/kaggle-tgs-salt/kfold_list/cv4')


def submit(df, filename):
    filepath = str(submission_dir / filename)
    df.drop(['z'], axis=1).to_csv(filepath, index=False)


def generate_kfold_list(n_cv=4, seed=1234):
    for k, (train_df, val_df) in enumerate(kfold_dfs(n_cv=n_cv, seed=seed)):
        train_df.to_csv(str(kfold_list_dir / f'train_cv{k}.csv'), index=False)
        val_df.to_csv(str(kfold_list_dir / f'val_cv{k}.csv'), index=False)


def get_train_df_fold(k):
    train_df = pd.read_csv(str(kfold_list_dir / f'train_cv{k}.csv'))
    return train_df


def get_val_df_fold(k):
    val_df = pd.read_csv(str(kfold_list_dir / f'val_cv{k}.csv'))
    return val_df


def get_dfs_fold(k):
    train_df = get_train_df_fold(k)
    val_df = get_val_df_fold(k)
    return train_df, val_df


def kfold_dfs(n_cv=4, seed=1234):
    train_df = pd.read_csv(train_csv)
    depth_df = pd.read_csv(depth_csv)

    # Append depth data
    train_df = pd.merge(train_df, depth_df, on='id')

    # Append coverage data
    n_pixels = 101 * 101
    def _cov(mask):
        return mask.sum() / n_pixels
    train_df['mask'] = [np.array(get_train_mask(_id), dtype=np.uint8) / 255
                        for _id in train_df.id] # dtype=np.float64
    train_df['coverage'] = train_df['mask'].apply(_cov)
    train_df['coverage_class'] = train_df['mask'].apply(get_mask_type)
    train_df = train_df.drop('mask', axis=1)

    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    for train_index, val_index in skf.split(train_df.index.values, train_df.coverage_class):
        fold_train_df = train_df.iloc[train_index]
        fold_val_df = train_df.iloc[val_index]
        yield fold_train_df, fold_val_df


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


def evaluate(model, loader, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    predictions = np.empty((0, 1, 101, 101), dtype=np.int8)
    targets = np.empty((0, 1, 101, 101), dtype=np.int8)
    for data, target in loader:
        pred = predict(model, data, **kwargs)
        target = target.cpu().detach().numpy().astype(np.int8)

        predictions = np.concatenate([predictions, pred])
        targets = np.concatenate([targets, target])

    #return jaccard_similarity_score(targets.flatten(), predictions.flatten())
    return get_iou_vector(targets, predictions)


def predict(model, x, device=None,
            use_sigmoid=True, threshold=0.5,
            with_tta=False):
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((101, 101))

    x = x.to(device)

    if with_tta:
        logits = tta(model, x)
    else:
        logits = model(x)

    if use_sigmoid:
        pred = torch.sigmoid(logits)
    else:
        pred = logits

    pred = (pred > threshold).cpu()
    pred = np.array([np.array(resize(to_pil(_))) for _ in pred])
    pred = pred.reshape(pred.shape[0], 1, 101, 101).astype(np.float32)
    return torch.Tensor(pred)


def predict_kfold_cv(models, x, device=None,
                     use_sigmoid=True, threshold=0.5,
                     with_tta=False):
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((101, 101))

    accum_pred = torch.Tensor(np.zeros((x.shape[0], 1, x.shape[2], x.shape[3])))
    x = x.to(device)

    for model in models:
        if with_tta:
            logits = tta(model, x)
        else:
            logits = model(x)

        if use_sigmoid:
            pred = torch.sigmoid(logits)
        else:
            pred = logits

        accum_pred += pred.cpu()
    
    accum_pred = accum_pred / len(models)
    accum_pred = (accum_pred > threshold)
    accum_pred = np.array([np.array(resize(to_pil(_))) for _ in accum_pred])
    accum_pred = accum_pred.reshape(accum_pred.shape[0], 1, 101, 101).astype(np.float32)
    return torch.Tensor(accum_pred)


def tta(model, x):
    """Test time augmentation
    """
    logits = model(x)
    x = x.flip(3)
    logits += model(x).flip(3)
    return 0.5 * logits


def get_iou_vector(A, B):
    """
    A: gt
    B: pred
    shape: (n, c, x, y)
    """
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return ((intersection + epsilon)/ (union - intersection + epsilon)).mean()


def save_model(model, keyname):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))
    
    filename = f'{keyname}.model'
    torch.save(model, str(model_dir / filename))


def add_depth_channels(image_tensor):
    """https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949#385778
    """
    _, h, w = image_tensor.size()
    im = np.zeros((3, h, w)).astype(np.float32)
    im[0, :, :] = image_tensor.numpy()[0]
    for row, const in enumerate(np.linspace(0, 1, h)):
        im[1, row, :] = const
    im[2] = im[0] * im[1]
    return torch.Tensor(im)


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


def get_mask_type(mask):
    """Reference  from Heng's discussion
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
    """
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border,
                               borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1 #border
    if np.all(mask == mask[0]):
        return 2 #vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1, 8))
    for c in coverage:
        histall[0, c] += 1
    return histall