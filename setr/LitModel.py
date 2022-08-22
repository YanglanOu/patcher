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
from torch_poly_lr_decay import PolynomialLRDecay
# from setr.SETR import SETR_Naive_S_CS
from utils import *

from mmseg.models import build_segmentor
from mmcv.runner import build_optimizer

class LitSETR(pl.LightningModule):
    def __init__(self, cfg, batch_size=-1):
        super(LitSETR, self).__init__()
        model = build_segmentor(
            cfg.cfg_mm.model, train_cfg=cfg.cfg_mm.train_cfg, test_cfg=cfg.cfg_mm.test_cfg)

        self.register_buffer('batch_size', torch.tensor(batch_size))
        self.model = model
        self.lr_scheduler = True
        self.cfg = cfg
        self.preds = []
        self.epoch = 0

    def training_step(self, batch, batch_idx):
        batch['img_metas'] = batch['img_metas'].data[0]
        batch['img'] = batch['img'].data[0].to(self.device)
        batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)
        outputs = self.model.forward_train(**batch)

        if 'aux_0.loss_seg' in outputs:
            loss = outputs['decode.loss_seg'] + outputs['aux_0.loss_seg'] + outputs['aux_1.loss_seg'] + outputs['aux_2.loss_seg']
        elif 'aux.loss_seg' in outputs:
            loss = outputs['decode.loss_seg'] + outputs['aux.loss_seg']
        else:
            loss = outputs['decode.loss_seg']
            
        acc = outputs['decode.acc_seg']
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch['img_metas'] = batch['img_metas'].data[0]
        # batch['img'] = batch['img'].data[0].to(self.device)
        # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)
        # outputs = self.model.val_step(batch)

        labels = batch.pop('gt_semantic_seg')
        labels = labels[0].cpu().numpy()

        batch['img_metas'][0] = batch['img_metas'][0].data[0]
        batch['rescale'] = False
        preds = self.model(return_loss=False, **batch)
        preds = np.array(preds)

        dice = dice_coeff(preds, labels)

        self.log('val_dice', dice, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        batch['img_metas'][0] = batch['img_metas'][0].data[0]
        # batch['img'][0] = batch['img'][0].data[0].to(self.device)
        outputs = self.model(return_loss=False, **batch)

        return outputs

    def test_step_end(self, output):
        self.preds.extend(output)

    def get_preds(self):
        return self.preds
    
    def on_load_checkpoint(self, checkpoint):
        self.epoch = checkpoint['epoch']

    def get_epoch(self):
        return self.epoch

    def configure_optimizers(self):
        if self.cfg.use_mmcv_optim:
            optimizer = build_optimizer(self.model, self.cfg.cfg_mm.optimizer)
        elif self.cfg.optim_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-8, momentum=0.9)

        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=self.cfg.max_decay_epochs, end_learning_rate=self.cfg.end_lr, power=0.9)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    

        # optimizer = optim.RMSprop(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-8, momentum=0.9)
        # return {
        #     'optimizer': optimizer
        #     # 'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5),
        #     # 'lr_scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1),
        #     'lr_scheduler': get_scheduler(optimizer, 'lambda', nepoch_fix=20, nepoch=self.cfg.max_decay_epochs),
        #     'monitor': 'val_dice'
        # }