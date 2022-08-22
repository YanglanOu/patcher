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

# indx = list(range(32))
# rs.shuffle(indx)
# # fold = 2
# training_cases = []
# train_file = open('training_cases.txt', "r")
# lines = train_file.readlines()
# data_dir = os.path.expanduser('~/data/Stroke_AWS_DWI_Anom_PSU/')

# for line in lines:
#     if line != '\n':
#         line = line.split('/')
#         case = data_dir + line[1] + '/' + line[2]
#         training_cases.append(case)

# all_files = glob.glob(os.path.join(data_dir, '*/*'))
# all_files.sort()
# val_list = list(set(all_files) - set(training_cases))
# val_list.sort()

# for fold in [0,1,2]:
#     if fold == 0:
#         val_indx = indx[:20]
#         test_indx = indx[20:]
#     elif fold == 1:
#         val_indx = indx[:10] + indx[20:]
#         test_indx = indx[10:20]
#     elif fold == 2:
#         val_indx = indx[10:]
#         test_indx = indx[:10]

#     for mode in ['val', 'test']:
#         if mode == 'val':
#             ind = val_indx
#         else:
#             ind = test_indx

#         img_dir = f'/data/stroke_trans/fold_{fold}/img_dir/{mode}'
#         ann_dir = f'/data/stroke_trans/fold_{fold}/ann_dir/{mode}'

#         for case in [val_list[i] for i in ind]:
#             line = case.split('/')

#             eadc_file = case + '/eADC.nii.gz'
#             eadc = nib.load(eadc_file).get_fdata()

#             slices = eadc.shape[-1]

#             dwi_file = case + '/rb1000.nii.gz'
#             dwi = nib.load(dwi_file).get_fdata()

#             label_file = case + '/rb1000_roi.nii.gz'
#             label = nib.load(label_file).get_fdata()

#             for i in range(slices):
#                 file_name = f'{line[-2]}_{line[-1]}_{i}.npy'
#                 eadc_i = eadc[...,i]
#                 dwi_i = dwi[...,i]
#                 input = np.stack((eadc_i, dwi_i), axis=-1)

#                 label_i = label[...,i]

#                 np.save(f'{img_dir}/{file_name}', input)
#                 print(f'{img_dir}/{file_name}')
#                 np.save(f'{ann_dir}/{file_name}', label_i)

# exit(0)

# # training data    
# file = open('training_cases.txt', "r")
# lines = file.readlines()
# eadcs = []
# dwis = []
# labels = []
# data_dir = os.path.expanduser('~/data/Stroke_AWS_DWI_Anom_PSU/')

# mode = 'train'
# for fold in [0,1,2]:
#     img_dir = f'/data/stroke_trans/fold_{fold}/img_dir/{mode}'
#     ann_dir = f'/data/stroke_trans/fold_{fold}/ann_dir/{mode}'

#     for line in lines:
#         if line != '\n':
#             line = line.split('/')
#             case = data_dir + line[1] + '/' + line[2]

#             eadc_file = case + '/eADC.nii.gz'
#             eadc = nib.load(eadc_file).get_fdata()

#             slices = eadc.shape[-1]

#             dwi_file = case + '/rb1000.nii.gz'
#             dwi = nib.load(dwi_file).get_fdata()

#             label_file = case + '/rb1000_roi.nii.gz'
#             label = nib.load(label_file).get_fdata()
            
#             for i in range(slices):
#                 file_name = f'{line[1]}_{line[2]}_{i}.npy'
#                 eadc_i = eadc[...,i]
#                 dwi_i = dwi[...,i]
#                 input = np.stack((eadc_i, dwi_i), axis=-1)

#                 label_i = label[...,i]

#                 np.save(f'{img_dir}/{file_name}', input)
#                 np.save(f'{ann_dir}/{file_name}', label_i)


# # Kvasir-SEG
# indx = list(range(1000))
# rs.shuffle(indx)

# train_ind = indx[:800]
# val_ind = indx[800:900]
# test_ind = indx[900:]

# img_dir = '/data/Kvasir-SEG/images'
# all_imgs = glob.glob(os.path.join(img_dir, '*'))

# phases = ['train', 'val', 'test']
# for phase in phases:
#     if phase == 'train':
#         subset = [all_imgs[i] for i in train_ind]
#     elif phase == 'val':
#         subset = [all_imgs[i] for i in val_ind]
#     else:
#         subset = [all_imgs[i] for i in test_ind]

#     for data in subset:
#         img = Image.open(data)
#         img = img.resize((512,512))
#         filename = data.split('/')[-1]
#         file_id= filename.split('.')[0]

#         ann = Image.open(f'/data/Kvasir-SEG/masks/{filename}').convert('L')
#         # ann = ann.resize((512,512))

#         ann_value = np.round(np.asarray(ann)/255)
    
#         ann = Image.fromarray(np.uint8(ann_value))
#         ann = ann.resize((512,512))
#         ann = np.asarray(ann, dtype=np.float32)
#         print(ann_value.max(), filename)

#         img_dest = f'/data/Kvasir-SEG/img_dir/{phase}/{filename}'
#         ann_dest = f'/data/Kvasir-SEG/ann_dir/{phase}/{file_id}.npy'
        
#         img.save(img_dest)
#         np.save(ann_dest, ann)
#         # shutil.copyfile(f"/data/Kvasir-SEG/images/{filename}", img_dest)
#         # shutil.copyfile(f"/data/Kvasir-SEG/masks/{filename}", ann_dest)

# pass

all_imgs = glob.glob(os.path.join('/data/polyp/ValDataset/img_dir', 'CVC-300*'))
# all_imgs = glob.glob(os.path.join('/data/Kvasir-SEG/img_dir/train', '*'))
ch_0s = []
ch_1s = []
ch_2s = []
for img_name in all_imgs:
    # img = mpimg.imread(img_name)
    img = Image.open(img_name)
    img = np.asarray(img)
    ch_0 = img[...,0]
    ch_0s.append(ch_0)
    ch_1 = img[...,1]
    ch_1s.append(ch_1)
    ch_2 = img[...,2]
    ch_2s.append(ch_2)

ch_0s = np.array(ch_0s)
ch_1s = np.array(ch_1s)
ch_2s = np.array(ch_2s)

print('ch 0')
print(ch_0s.max())
print(ch_0s.min())
print(ch_0s.mean())
print(ch_0s.std())

print('ch 1')
print(ch_1s.max())
print(ch_1s.min())
print(ch_1s.mean())
print(ch_1s.std())

print('ch 2')
print(ch_2s.max())
print(ch_2s.min())
print(ch_2s.mean())
print(ch_2s.std())
exit(0)

# polyp
all_test_dir = '/data/polyp/TestDataset/'
all_testsets = glob.glob(os.path.join(all_test_dir, '*'))

# testset_dir = '/data/polyp/TestDataset/Kvasir'
for testset_dir in all_testsets:

    all_imgs = glob.glob(os.path.join(f'{testset_dir}/images', '*'))

    for data in all_imgs:
        img = Image.open(data)
        img = img.resize((352,352))
        test_set = data.split('/')[-3]
        filename = data.split('/')[-1]
        file_id= filename.split('.')[0]

        ann = Image.open(f'{testset_dir}/masks/{filename}').convert('L')
        # ann = ann.resize((512,512))

        ann_value = np.round(np.asarray(ann)/255)

        ann = Image.fromarray(np.uint8(ann_value))
        ann = ann.resize((352,352))
        ann = np.asarray(ann, dtype=np.float32)
        print(ann_value.max(), filename)

        new_filename = f'{test_set}_{filename}'
        new_file_id = f'{test_set}_{file_id}'
        img_dest = f'/data/polyp/ValDataset/img_dir/{new_filename}'
        ann_dest = f'/data/polyp/ValDataset/ann_dir/{new_file_id}.npy'
        
        img.save(img_dest)
        np.save(ann_dest, ann)