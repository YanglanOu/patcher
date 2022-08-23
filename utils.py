import torch
from torch.autograd import Function
from torch.nn import functional as F
from math import floor
import numpy as np
from torch.optim import lr_scheduler
from einops import rearrange
import pandas as pd
import glob
import os.path as osp


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = np.dot(input.flatten(), target.flatten())
        self.union = np.sum(input) + np.sum(target) + eps

        t = (2 * self.inter + eps) / self.union
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    s = 0.0

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceCoeff_tr(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = np.dot(input.flatten(), target.flatten())
        self.union = np.sum(input) + np.sum(target) + eps

        if np.sum(input) > 0 and np.sum(target) == 0:
            return 1.0

        else: 
            t = (2 * self.inter + eps) / self.union
            return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff_tr(input, target):
    """Dice coeff for batches"""
    s = 0.0

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff_tr().forward(c[0], c[1])

    return s / (i + 1)



def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def overlay(label_i, pred_i):
    union = label_i * pred_i
    white = label_i - union
    red = pred_i - union

    white = np.stack((white,) * 3, axis=-1)
    white = white * [240, 240, 240]

    red = np.stack((red,) * 3, axis=-1)
    red = red * [102, 102, 255]

    union = np.stack((union,) * 3, axis=-1)
    union = union * [102, 204, 0]    

    res_fig = white + red + union

    return res_fig


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    p: patch_size
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = rearrange(x, 'b c h ph w pw -> (b h w) c ph pw')
    return x


def patchify_unfold(imgs, kernel_size, stride=1, padding=0, dilation=1):
    """
    imgs: (N, C, H, W)
    p: patch_size
    x: (N x h x w, C, P, P)
    """
    N, C, H, W = imgs.shape
    h = floor(((H + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    # w = floor(((W + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    x = F.unfold(imgs, kernel_size=(kernel_size, kernel_size), stride=stride, dilation=dilation, padding=padding)
    x = rearrange(x, 'b (c ph pw) (h w) -> (b h w) c ph pw', ph=kernel_size, pw=kernel_size, h=h)
    return x


def patchify_enlarged(imgs, patch_size, context_padding, padding_mode='replicate'):
    kernel_size = patch_size + context_padding * 2
    # patche2 = patchify_unfold(imgs, kernel_size=kernel_size, stride=patch_size, padding=context_padding)
    imgs_pad = F.pad(imgs, (context_padding, context_padding, context_padding, context_padding), mode=padding_mode)
    patches = patchify_unfold(imgs_pad, kernel_size=kernel_size, stride=patch_size)
    return patches


def remove_padding(imgs, padding):
    return imgs[..., padding:-padding, padding:-padding]


def unpatchify(x, batch_size, context_padding=0):
    """
    x: ((N h w), C, patch_size, patch_size)
    imgs: (N, C, H, W)
    """
    h = w = int((x.shape[0]/batch_size)**.5)
    assert h * w == x.shape[0]/batch_size
    
    if context_padding > 0:
        x = remove_padding(x, padding=context_padding)
    imgs = rearrange(x, '(b h w) c ph pw -> b c (h ph) (w pw)', h=h, w=w)
    return imgs


def write_metrics_to_csv(csv_file, cfg, mIoU, dice, IoU, version, epoch, batch_size, last_epoch):
    df = pd.read_csv(csv_file)
    if not ((df['config_id'] == cfg.id) & (df['fold'] == cfg.fold) & (df['epoch'] == epoch) & (df['batch_size'] == batch_size)).any():
        df = df.append({'config_id': cfg.id, 'fold': cfg.fold, 'epoch': epoch, 'batch_size': batch_size}, ignore_index=True)
    index = (df['config_id'] == cfg.id) & (df['fold'] == cfg.fold) & (df['epoch'] == epoch)
    
    df.loc[index, 'l_patch'] = cfg.cfg_mm.train_cfg['large_patch'] if 'large_patch' in cfg.cfg_mm.train_cfg else None
    df.loc[index, 's_patch'] = cfg.cfg_mm.train_cfg['small_patch']if 'small_patch' in cfg.cfg_mm.train_cfg else None
    df.loc[index, 'mIoU'] = mIoU
    df.loc[index, 'dice'] = dice  
    df.loc[index, 'IoU'] = IoU  
    df.loc[index, 'version'] = version  
    df.loc[index, 'epoch'] = epoch  
    df.loc[index, 'batch_size'] = batch_size  
    df.loc[index, 'last_epoch'] = last_epoch  

    df.to_csv(csv_file, index=False, float_format='%f')


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

def find_last_version(folder, prefix='version_'):
    version_folders = glob.glob(f'{folder}/{prefix}*')
    version_numbers = sorted([int(osp.basename(x)[len(prefix):]) for x in version_folders])
    last_version = version_numbers[-1]
    return last_version