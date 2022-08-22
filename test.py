import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Cityscapes
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
from numpy.random import RandomState
from mmseg.datasets import build_dataloader, build_dataset

from config import Config
from utils import *
from setr.LitModel import *
from unet.LitUNet import LitUNet


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', default='test')
    parser.add_argument('--gpus', default='2,5,6')
    parser.add_argument('--version', default=None)
    parser.add_argument('--model', default='setr')
    parser.add_argument('--vis', action='store_true', default=False)
                 
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.cfg)

    rs = RandomState(cfg.seed)
    gpus = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpus)
    # if num_gpus > 1:
    #     cfg.batchsize //= num_gpus

    seed_everything(cfg.seed, workers=True)

    test_dataset = build_dataset(cfg.cfg_mm.data.test)
    testloader = build_dataloader(
                    test_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.cfg_mm.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)

    
    if args.version is None:
        args.version = find_last_version(f'{cfg.cfg_dir}/lightning_logs/')    

    if cfg.baseline_model is not None:
        net = LitUNet.load_from_checkpoint(f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/setr-best.ckpt', cfg=cfg, model=cfg.baseline_model, strict=False)
        net_last = LitUNet.load_from_checkpoint(f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/last.ckpt', cfg=cfg, model=cfg.baseline_model, strict=False)
    else:
        # net = LitSETR.load_from_checkpoint(f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/best_8832.ckpt', cfg=cfg, strict=False)
        net = LitSETR.load_from_checkpoint(f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/setr-best.ckpt', cfg=cfg, strict=False)
        net_last = LitSETR.load_from_checkpoint(f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/last.ckpt', cfg=cfg, strict=False)

    epoch = net.get_epoch()
    last_epoch = net_last.get_epoch()

    trainer = pl.Trainer(gpus=gpus, default_root_dir=cfg.test_dir)
    log = trainer.test(net, test_dataloaders=testloader)
    preds = net.get_preds()


    bs = net.batch_size.numpy()

    res = test_dataset.evaluate(preds, 'mIoU', vis=args.vis, save_dir=cfg.vis_dir)

    if cfg.dataset == 'polyp':
        csv_file = 'results_csv/polyp.csv'
    elif cfg.dataset == 'kvasir':
        csv_file = 'results_csv/kvasir.csv'
    else:
        csv_file = 'results_csv/stroke.csv'

    write_metrics_to_csv(csv_file, cfg, res['mIoU'], res['dice'], res['stroke_IoU'], version=args.version, epoch=epoch, batch_size=bs, last_epoch=last_epoch)
    print(res)
