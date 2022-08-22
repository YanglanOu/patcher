import argparse
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch 
import torchvision.transforms.functional as TF
import random
import copy
from mmseg.datasets import build_dataloader, build_dataset

from config import Config
from numpy.random import RandomState
from setr.LitModel import *
from torchvision import transforms

from unet.LitUNet import LitUNet
# from dataset import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='test')
    parser.add_argument('--gpus', default='4')
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--fold', default=0)
    parser.add_argument('--save_n_epochs', default=50)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--version', default=0)

    args = parser.parse_args()
    cfg = Config(args.cfg, tmp=args.tmp, create_dirs=False)
    # cfg1 = Config1.fromfile('configs/SETR/mysetr_depth18.py')
    rs = RandomState(cfg.seed)

    seed_everything(cfg.seed, workers=True)
    gpus = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpus)
    # if num_gpus > 1:
    #     cfg.batchsize //= num_gpus


    train_dataset = build_dataset(cfg.cfg_mm.data.train)
    trainloader = build_dataloader(
                    train_dataset,
                    cfg.cfg_mm.data.samples_per_gpu,
                    cfg.cfg_mm.data.workers_per_gpu,
                    # cfg.gpus will be ignored if distributed
                    1,
                    dist=False,
                    seed=cfg.seed,
                    drop_last=True)
    # print(train_dataset[0])
    # print(train_dataset[0])

    val_dataset = copy.deepcopy(cfg.cfg_mm.data.val)
    # val_dataset.pipeline = cfg.cfg_mm.data.train.pipeline
    val_dataset = build_dataset(val_dataset)
    valloader = build_dataloader(
                    val_dataset,
                    cfg.cfg_mm.data.samples_per_gpu,
                    cfg.cfg_mm.data.workers_per_gpu,
                    # cfg.gpus will be ignored if distributed
                    1,
                    dist=False,
                    seed=cfg.seed,
                    drop_last=True)
    test_dataset = build_dataset(cfg.cfg_mm.data.test)
    testloader = build_dataloader(
                    test_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.cfg_mm.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)

    if cfg.baseline_model is not None:
        setr = LitUNet(cfg, cfg.baseline_model, batch_size=cfg.cfg_mm.data.samples_per_gpu * num_gpus)
    else:
        setr = LitSETR(cfg, batch_size=cfg.cfg_mm.data.samples_per_gpu * num_gpus)
    

    checkpoint_callback = ModelCheckpoint(
        # dirpath=cfg.cfg_dir,
        monitor='val_dice',
        filename='setr-best',
        save_last=True,
        save_top_k=1,
        mode='max',
    ) 

    checkpoint_epoch_cb = ModelCheckpoint(
        monitor='val_dice',
        # dirpath=checkpoint_dir,
        filename='model-{epoch:04d}',
        save_last=False,
        save_top_k=-1,
        mode='max',
        every_n_val_epochs=args.save_n_epochs
    )

    trainer = pl.Trainer(
        gpus=gpus,
        auto_select_gpus=True,
        max_epochs=cfg.epochs,
        resume_from_checkpoint=f'{cfg.cfg_dir}/lightning_logs/version_{args.version}/checkpoints/last.ckpt' if args.resume else None,
        default_root_dir=cfg.cfg_dir,
        accelerator='ddp',  # if num_gpus > 1 else None,
        callbacks=[checkpoint_callback, checkpoint_epoch_cb]
    )
    trainer.fit(setr, trainloader, valloader)
    trainer.test(test_dataloaders=testloader, ckpt_path='best')
    preds = setr.get_preds()

    res = test_dataset.evaluate(preds, 'mIoU')