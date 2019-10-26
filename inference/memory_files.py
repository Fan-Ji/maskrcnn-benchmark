from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list
from torch.utils.data import Dataset
import math
import torch
class MemoryFilesCollator(object):
    def __init__(self, divisibility):
        self.divisibility = divisibility
    def __call__(self, images):
        return to_image_list(images, self.divisibility)

class MemoryFiles(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is None:
            return img
        for t in self.transforms:
            img = t(img, None)
            if isinstance(img, tuple):
                img = img[0]
        return img

    def __len__(self):
        return len(self.images)
