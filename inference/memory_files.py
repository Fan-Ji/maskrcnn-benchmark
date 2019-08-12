from torch.utils.data import Dataset
import math
import torch
class MemoryFiles(Dataset):
    def __init__(self, images, transforms, divisibility=0):
        self.divisibility = divisibility
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.images[index]
        for t in self.transforms:
            img = t(img, None)
            if isinstance(img, tuple):
                img = img[0]
        #TODO pre-processing to improve efficiency??
        if self.divisibility:
            stride = self.divisibility
            img_sz = img.shape
            sz = list(img.shape)
            sz[1] = int(math.ceil(sz[1] / stride) * stride)
            sz[2] = int(math.ceil(sz[2] / stride) * stride)
            ret = torch.zeros(sz)
            ret[:img_sz[0], :img_sz[1], :img_sz[2]].copy_(img)
            return ret
        return img

    def __len__(self):
        return len(self.images)
