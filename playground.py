import torch
from torch.nn import functional as F
from einops import rearrange
from math import floor


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
    x: ((N h w), C, patch_size**2)
    imgs: (N, C, H, W)
    """
    h = w = int((x.shape[0]/batch_size)**.5)
    assert h * w == x.shape[0]/batch_size
    
    if context_padding > 0:
        x = remove_padding(x, padding=context_padding)
    imgs = rearrange(x, '(b h w) c ph pw -> b c (h ph) (w pw)', h=h, w=w)
    return imgs


x = torch.arange(25).float().reshape(5, 5)
print(x)
x = x[None, None, ...].repeat(1, 3, 1, 1)
print(x.shape)
y = torch.nn.functional.unfold(x, (4, 4))
ys = y.view(1, 3, 4, 4, -1)
print(ys[0, ..., 0])
print(ys[0, ..., 1])

imgs = torch.arange(36).float().reshape(6, 6)
imgs = imgs[None, None, ...].repeat(1, 3, 1, 1)
patch = patchify_unfold(imgs, 2, 2)
patch2 = patchify(imgs, 2)
diff = patch - patch2
patch3 = patchify_enlarged(imgs, 2, context_padding=1)
# network processing
# patch_orig = remove_padding(patch3, padding=1)
img_recon = unpatchify(patch3, 1, context_padding=1)
print(diff)
print(patch)