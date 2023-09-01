"""Module containing the dataset related functionality."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import csv, os
from transoar.data.transforms import get_transforms
from utils_selfv1 import *

def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # 跳过第一行

        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


def get_filenames(path):
    filenames = []
    for filename in os.listdir(path):
        filenames.append(filename.split('_')[0])
        # embed()
    return filenames

class TransoarDataset(Dataset):
    """Dataset class of the transoar project."""
    def __init__(self, 
                 mode='train',
                 root_dir='',
                 data_process='crop',
                 crop_size=(256, 256, 256),
                 debug=False,
                 ):
  

        super(TransoarDataset, self).__init__()
        self.mode=mode
        self.root_dir=root_dir
        self.data_process=data_process
        self.crop_size=crop_size
        self.debug=debug
        self.setup()


    def __getitem__(self, index):

        name = self.names[index]
        # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.129567032250534530765928856531'
        # time_1 = time()
        if self.data_process == 'resize':
            # dict = resize_data(name, self.root_dir, new_shape=(512, 512, 256))
            pass
        elif self.data_process == 'crop':
            # dict = crop_data(name, self.root_dir, new_shape=(256, 256, 256))
            dict = random_crop_3d(name, crop_size=self.crop_size, p=0.8, augmentatoin=False)
        # print('image_crop : {}'.format(time() - time_1))

        return dict
       
    
    

    def __len__(self):

        return len(self.names)
        
    def setup(self):
        print('set up ')
        if self.mode == 'train':
            file_path = f'/public_bme/data/xiongjl/det/csv_file/names.csv'
        elif self.mode == 'valid':
            file_path = '/public_bme/data/xiongjl/det/csv_file/test_names.csv'
        # print(file_path)
        self.names = read_names_from_csv(file_path)

        if self.mode == 'train':
            random.seed(1)
            random.shuffle(self.names)