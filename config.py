import yaml
import os
import numpy as np
import shutil
import glob
from mmcv.utils import Config as Config1


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


class Config:
    def __init__(self, cfg_id, tmp=False, create_dirs=False):
        self.id = cfg_id
        cfg_path = 'configs/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        with open(files[0], 'r') as stream:
            cfg = yaml.safe_load(stream)
    
        self.cfg_mm = Config1.fromfile(cfg['cfg_file'])

        # res dir for pytorch lighting 
        self.base_dir = '/tmp/transformer' if tmp else cfg.get('results_dir', 'results/')
        self.cfg_dir = f'{self.base_dir}/{cfg_id}' 
        self.test_dir = f'{self.cfg_dir}/test' 
        self.vis_dir = f'{self.cfg_dir}/vis' 

        # if create_dirs == True:
        #     if os.path.exists(self.cfg_dir):
        #         shutil.rmtree(self.cfg_dir)

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.seed = cfg['seed']
        
        self.baseline_model = cfg.get('baseline_model', None)
        self.dataset = cfg['dataset']
        self.n_channels = cfg.get('n_channels', 2)
        self.n_classes = cfg.get('n_classes', 2)
        self.trans_Unet = cfg.get('trans_Unet', None)
        
        self.lr = cfg['lr']
        self.end_lr = cfg.get('end_lr', 0.0001)
        self.epochs = cfg['epochs']
        self.max_decay_epochs = cfg.get('max_decay_epochs', self.epochs)
        self.use_mmcv_optim = cfg.get('use_mmcv_optim', False)
        self.optim_type = cfg.get('optim_type', 'SGD')
        
        self.fold = cfg['fold']
        if self.dataset == 'stroke':
            
            self.cfg_mm.data.train.data_root = os.path.expanduser(f'~/data/stroke_trans/fold_{self.fold}')
            self.cfg_mm.data.val.data_root = os.path.expanduser(f'~/data/stroke_trans/fold_{self.fold}')
            self.cfg_mm.data.test.data_root = os.path.expanduser(f'~/data/stroke_trans/fold_{self.fold}')