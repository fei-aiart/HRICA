from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms.transforms as T
import matplotlib.pyplot as plt
import numpy as np


class datasetLoader(Dataset):
    def __init__(self, images_path: list, ann_path: list, transform=None):
        self.images_path = images_path
        self.ann_path = ann_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        ann = plt.imread(self.ann_path[item])
        ann = np.array(ann * 255)
        ann = Image.fromarray(ann)
        ann = T.Resize(512)(ann)
        ann = T.ToTensor()(ann)
       
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.transform(img)

        return img, ann