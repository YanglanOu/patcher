import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2
import glob
from torch.utils import data
from numpy.random import RandomState
import shutil
import matplotlib.image as mpimg
from PIL import Image

rs = RandomState(523)

# Kvasir-SEG
indx = list(range(1000))
rs.shuffle(indx)

train_ind = indx[:800]
val_ind = indx[800:900]
test_ind = indx[900:]

img_dir = 'data/Kvasir-SEG/images'
all_imgs = glob.glob(os.path.join(img_dir, '*'))

phases = ['train', 'val', 'test']
for phase in phases:
    if phase == 'train':
        subset = [all_imgs[i] for i in train_ind]
    elif phase == 'val':
        subset = [all_imgs[i] for i in val_ind]
    else:
        subset = [all_imgs[i] for i in test_ind]

    for data in subset:
        img = Image.open(data)
        img = img.resize((512,512))
        filename = data.split('/')[-1]
        file_id= filename.split('.')[0]

        ann = Image.open(f'data/Kvasir-SEG/masks/{filename}').convert('L')

        ann_value = np.round(np.asarray(ann)/255)
    
        ann = Image.fromarray(np.uint8(ann_value))
        ann = ann.resize((512,512))
        ann = np.asarray(ann, dtype=np.float32)
        print(filename)

        img_dest = f'data/Kvasir-SEG/img_dir/{phase}/{filename}'
        ann_dest = f'data/Kvasir-SEG/ann_dir/{phase}/{file_id}.npy'
        
        img.save(img_dest)
        np.save(ann_dest, ann)
        # shutil.copyfile(f"/data/Kvasir-SEG/images/{filename}", img_dest)
        # shutil.copyfile(f"/data/Kvasir-SEG/masks/{filename}", ann_dest)

