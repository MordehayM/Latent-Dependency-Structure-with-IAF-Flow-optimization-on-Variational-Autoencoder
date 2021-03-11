import numpy as np
import torch, os
from base import BaseDataLoader
import mnist
from torchvision import datasets, transforms


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):

        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        self.data_dir = data_dir
      
 
        self.dataset = BinarizedMNISTDataset( self.data_dir, train=training)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BinarizedMNISTDataset(torch.utils.data.Dataset):
    """
    MNIST dataset converted to binary values.
    """

    def __init__(self, data_dir, train=True):
        
            
        # if train:

        #     self.datasets = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        #     self.images, self.labels = dataset.data, dataset.targets
        # else:

        #     self.datasets = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        #     self.images, self.labels = dataset.data, dataset.targets

        x_train, t_train, x_test, t_test = mnist.load()
        if train:
            self.images = torch.from_numpy(x_train)
            self.labels = torch.from_numpy(t_train) 
        else:
            self.images = torch.from_numpy(x_test)
            self.labels = torch.from_numpy(t_test)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx]
        labels = self.labels
        min_v = x.min()
        range_v = x.max() - min_v
        if range_v > 0:
            normalised = (x - min_v) / range_v
        else:
            normalised = torch.zeros(x.size()).to(x.device)
        return (normalised > 0.5).type(torch.float) #, labels