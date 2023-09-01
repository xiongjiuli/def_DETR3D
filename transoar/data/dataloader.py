"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox


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

# def init_fn(worker_id):
#     """
#     https://github.com/pytorch/pytorch/issues/7068
#     https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
#     """
#     torch_seed = torch.initial_seed()
#     if torch_seed >= 2**30:
#         torch_seed = torch_seed % 2**30
#     seed = torch_seed + worker_id

#     random.seed(seed)   
#     np.random.seed(seed)
#     monai.utils.set_determinism(seed=seed)
#     torch.manual_seed(seed)


class TransoarCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        for image, label in batch:
            batch_images.append(image)
            batch_labels.append(label)
            batch_masks.append(torch.zeros_like(image))

        # Generate bboxes and corresponding class labels
        batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)
        # 这个是从seg图中分离出bbox，形式是cxcyczwhd， 形式就是[[], [], []],但是里面是tensor
        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)
        # image, mask是全0的数值， labels就是seg图，然后 [(bbox1, class1), (bbox2, class2), ...]
        #* 这个才是最后传出来的东西，分为四个部分

