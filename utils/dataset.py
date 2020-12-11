# -*- coding:utf-8 -*-

from torch.utils.data.dataset import Dataset


class ClassificationDataSet(Dataset):

    def __init__(self, dataset_x, dataset_y):
        self.text = dataset_x
        self.label = dataset_y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]
