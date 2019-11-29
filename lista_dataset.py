import os
import torch
from torch.utils.data import Dataset
import numpy as np

class LISTADataset(Dataset):
    """LISTADataset"""
    def __init__(self, prob_dir = "/Users/wangjinxin/Desktop/LISTA/data", train=True):
        self._train = train
        self.A = np.load(os.path.join(prob_dir, 'Amatrix.npy'))
        self._ytrain = np.load(os.path.join(prob_dir, 'ytrain.npy'))
        self._xtrain = np.load(os.path.join(prob_dir, 'xtrain.npy'))
        self._ytest = np.load(os.path.join(prob_dir, 'ytest.npy'))
        self._xtest = np.load(os.path.join(prob_dir, 'xtest.npy'))

    def __len__(self):
        if self._train:
            return self._ytrain.shape[1]
        else:
            return self._ytest.shape[1]

    def __getitem__(self, index):
        if self._train:
            return torch.tensor(self._ytrain[:, index], dtype=torch.float32), \
                   torch.tensor(self._xtrain[:, index], dtype=torch.float32)
        else:
            return torch.tensor(self._ytest[:, index], dtype=torch.float32), \
                   torch.tensor(self._xtest[:, index], dtype=torch.float32)


if __name__ == '__main__':
    dataset = LISTADataset(train=True)
    from IPython import embed
    embed()