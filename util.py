import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def show_prediction(im, pred, target=None):
    to_pil = transforms.ToPILImage()

    assert im.ndimension() in (3, 4)
    assert pred.ndimension() == 4

    batch_size, _, size, _ = pred.shape

    if im.ndimension() == 3:
        im = im.reshape(1, *im.shape)
    assert im.shape[0] == batch_size


    if target is not None:
        if target is not None and target.ndimension() == 3:
            target = target.reshape(1, *target.shape)
        assert target.shape[0] == batch_size

        comb_img = None
        for _im, _pred, _target in zip(im, pred, target):
            row_img = np.concatenate([_im.numpy(), _pred.numpy(), _target.numpy()], axis=2)
            if comb_img is None:
                comb_img = row_img
            else:
                comb_img = np.concatenate([comb_img, row_img], axis=1)
        return to_pil(torch.Tensor(comb_img))

    # TODO
    comb_img = np.hstack([im.numpy(), pred.numpy()])
    return to_pil(comb_img)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
