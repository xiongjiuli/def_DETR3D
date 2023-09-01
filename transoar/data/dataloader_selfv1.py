"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset_selfv1 import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox, xyzwhdbbox


def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config)
    shuffle = False if split in ['test', 'val'] else config['shuffle']

    dataset = TransoarDataset(config, split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], collate_fn=collator
    )
    return dataloader


class TransoarCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        batch_xyz = []
        batch_whd = []
        for dct in batch:
            image = dct['input'] 
            croped_xyz = dct['new_coords'] 
            name = dct['name'] 
            croped_whd = dct['origin_whd'] 
            batch_images.append(image)
            # batch_labels.append(label)
            batch_masks.append(torch.zeros_like(image))
            batch_xyz.append(croped_xyz)
            batch_whd.append(croped_whd)

        # Generate bboxes and corresponding class labels
        # batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)  # * 从这里看出 0 是背景类
        batch_bboxes, batch_classes = xyzwhdbbox(batch_xyz, batch_whd)
        # 这个是从seg图中分离出bbox，形式是cxcyczwhd， 形式就是[[], [], []],但是里面是tensor
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)) # , torch.stack(batch_labels)
        # image, mask是全0的数值， labels就是seg图，然后 [(bbox1, class1), (bbox2, class2), ...]
        #* 这个才是最后传出来的东西，分为四个部分

