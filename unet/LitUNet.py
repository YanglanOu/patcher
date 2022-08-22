import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch import optim
import torch
import pytorch_lightning as pl
import numpy as np
import os
from torchmetrics import IoU
from torchmetrics.functional import accuracy
# from setr.SETR import *
# from setr.SETR import SETR_Naive_S_CS
from utils import *
from .unet_model import UNet, AttnUNet
from unet.trans_unet import VisionTransformer, TRANSCONFIG

from mmseg.models import build_segmentor


class LitUNet(pl.LightningModule):
    def __init__(self, cfg, model, batch_size=-1):
        super(LitUNet, self).__init__()
        if model == 'unet':
            self.model = UNet(cfg)
        elif  model == 'attnunet':
            self.model = AttnUNet(cfg)
        elif  model == 'transunet':
            assert cfg.trans_Unet is not None
            trans_config = TRANSCONFIG[cfg.trans_Unet]
            self.model = VisionTransformer(trans_config, img_size=256, num_classes=1)
        else:
            raise ValueError('unknow unet model!')

        self.register_buffer('batch_size', torch.tensor(batch_size))

        self.criterion = nn.BCEWithLogitsLoss()
        self.out = nn.Sigmoid()
        self.cfg = cfg
        self.preds = []

    def training_step(self, batch, batch_idx):
        batch['img_metas'] = batch['img_metas'].data[0]
        imgs = batch['img'].data[0].to(self.device)
        true_masks = batch['gt_semantic_seg'].data[0].to(self.device).float()
        # true_masks = torch.unsqueeze(true_masks, 1)

        
        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()

        true_masks = true_masks.clone()
        true_masks[true_masks == 255] = 0

        loss = self.criterion(masks_pred, true_masks)
        dice = dice_coeff(masks_pred_b.cpu().numpy(), true_masks.cpu().numpy())

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True)  
        return loss 

    def validation_step(self, batch, batch_idx):
        # batch['img_metas'] = batch['img_metas'].data[0]
        # batch['img'] = batch['img'].data[0].to(self.device)
        # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)
        # outputs = self.model.val_step(batch)
        imgs = batch['img'][0]
        true_masks = batch['gt_semantic_seg'][0]
        true_masks = torch.unsqueeze(true_masks, 1)
        # labels = labels[0].cpu().numpy()

        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()
        loss = self.criterion(masks_pred, true_masks)
        dice = dice_coeff(masks_pred_b.cpu().numpy(), true_masks.cpu().numpy())

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True)     

    def test_step(self, batch, batch_idx):
        imgs = batch['img'][0]
        # true_masks = batch['gt_semantic_seg'][0].long()
        true_masks = batch.pop('gt_semantic_seg')[0]

        true_masks = torch.unsqueeze(true_masks, 1)
        # labels = labels[0].cpu().numpy()

        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()
        loss = 0    # self.criterion(masks_pred, true_masks)
        dice = 0    # dice_coeff(masks_pred_b.cpu().numpy(), true_masks.cpu().numpy())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True)  
        
        return np.squeeze(masks_pred_b.cpu().numpy(), 1)

    def test_step_end(self, output):
        self.preds.extend(output)

    def get_preds(self):
        return self.preds
    
    def on_load_checkpoint(self, checkpoint):
        self.epoch = checkpoint['epoch']

    def get_epoch(self):
        return self.epoch

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5),
            # 'lr_scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1),
            'lr_scheduler': get_scheduler(optimizer, 'lambda', nepoch_fix=20, nepoch=200),
            'monitor': 'val_dice'
        }