import argparse
import glob
import os
import sys
import numpy as np
import ruamel.yaml
from copy import deepcopy


cfg_name = 'swin_stroke'
folds = [1,2]

files = glob.glob(f'configs/**/{cfg_name}.yml', recursive=True)
assert(len(files) == 1)
cfg_file = files[0]
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yml_orig = yaml.load(open(cfg_file, 'r'))

for fold in folds:
    new_cfg_file = cfg_file.replace(f'{cfg_name}', f'{cfg_name}_fold{fold}')
    print(cfg_file, new_cfg_file)
    if os.path.exists(new_cfg_file):
        print(f'cfg already exists {new_cfg_file}')
        continue
    yml = deepcopy(yml_orig)
    yml['fold'] = fold
    with open(new_cfg_file, 'w') as file:
        documents = yaml.dump(yml, file)