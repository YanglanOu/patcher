import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import glob
from torch.utils import data


class Stroke_train(Dataset):

    def __init__(self, data_dir, training_cases):
        file = open(training_cases, "r")
        lines = file.readlines()
        self.eadcs = []
        self.dwis = []
        self.labels = []
        self.case_names = []

        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]

                eadc_file = case + '/eADC.nii.gz'
                eadc = nib.load(eadc_file).get_fdata()

                slices = eadc.shape[-1]

                dwi_file = case + '/rb1000.nii.gz'
                dwi = nib.load(dwi_file).get_fdata()

                label_file = case + '/rb1000_roi.nii.gz'
                label = nib.load(label_file).get_fdata()

                for s in range(slices):
                    self.eadcs.append(eadc[..., s])
                    self.dwis.append(dwi[..., s])
                    self.labels.append(label[..., s])

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        eadc = self.eadcs[idx]
        eadc = eadc.astype(np.float32)
        eadc = np.expand_dims(eadc, axis=0)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.expand_dims(dwi, axis=0)
        
        input_data = np.concatenate((eadc, dwi), axis=0)

        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)

        return input_data, label


class Stroke_val(Dataset):

    def __init__(self, data_dir, training_files, indx):
        file = open(training_files, "r")
        lines = file.readlines()
        self.eadcs = []
        self.dwis = []
        self.labels = []
        training_cases = []
        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]
                training_cases.append(case)

        all_files = glob.glob(os.path.join(data_dir, '*/*'))
        all_files.sort()
        val_list = list(set(all_files) - set(training_cases))
        val_list.sort()
        for case in [val_list[i] for i in indx]:
            eadc_file = case + '/eADC.nii.gz'
            eadc = nib.load(eadc_file).get_fdata()

            slices = eadc.shape[-1]

            dwi_file = case + '/rb1000.nii.gz'
            dwi = nib.load(dwi_file).get_fdata()

            label_file = case + '/rb1000_roi.nii.gz'
            label = nib.load(label_file).get_fdata()

            case_name = case.split('/')
            case_name = case_name[-2] + '/' + case_name[-1]

            for s in range(slices):
                self.eadcs.append(eadc[..., s])
                self.dwis.append(dwi[..., s])
                self.labels.append(label[..., s])
                self.case_names.append(case_name)

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        eadc = self.eadcs[idx]
        eadc = eadc.astype(np.float32)
        eadc = np.expand_dims(eadc, axis=0)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.expand_dims(dwi, axis=0)
        
        input_data = np.concatenate((eadc, dwi), axis=0)

        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        
        name = self.case_names[idx]

        return input_data, label, name

if __name__ == '__main__':

    data_dir = '/home/yxo43/data/Stroke_AWS_DWI_Anom_PSU/'
    # data_dir = '/home/yxo43/data/stroke/Stroke_AWS_DWI_Anom_PSU_Partial_Truth/'
    training_cases = 'training_cases.txt'
    train_dataset = Stroke_train(data_dir, training_cases)
    trainloader = data.DataLoader(train_dataset,
                batch_size=1, shuffle=True, num_workers=0)
    eadcs = []
    dwis = []
    for batch in trainloader:
        imgs = batch[0]
        eadcs.append(imgs[0,0,...].numpy())
        dwis.append(imgs[0,1,...].numpy())
        
    eadcs = np.array(eadcs)
    dwis = np.array(dwis)
    print(eadcs.max())
    print(eadcs.min())
    print(eadcs.mean())
    print(eadcs.std())
    print(dwis.max())
    print(dwis.min())
    print(dwis.mean())
    print(dwis.std())
    pass