import torch
import torch.nn as nn
import numpy as np

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.datasets as dset



def cifar_to_gpu(cifar):
    """Loads cifar dataset to GPU memory. 
    Arguments:
        cifar (dataset): cifar dataset
    Output:
        cifar_gpu (dataset): cifar dataset in GPU memory
    """    
    loader = DataLoader(cifar, batch_size=512, num_workers=2)    
    y = torch.empty(0, dtype=torch.int64)
    x = torch.empty(0, dtype=torch.float32)
    for batch in loader:
        y = torch.cat((y, batch[1]), 0)
        x = torch.cat((x, batch[0]), 0)
    cifar_gpu = [x.type(torch.cuda.FloatTensor), 
                 y.type(torch.cuda.LongTensor)]
    return cifar_gpu


def get_loader_gpu(cifar_gpu, val_size=1024, batch_size=512):
    """Splits dataset into training set followed by validation set by specifying validation set size,
    training size = dataset size - validation size. Then from training set randomly subsamples
     the small training set (same size as validation set).
    
    Creates loader_gpu dictionary with three loaders.
    
    'val' (DataLoaderGPU): validation set

    'train' (DataLoaderGPU): training set

    'train_small' (DataLoaderGPU): small training set for accuracy checks

    Arguments:
        cifar_gpu (dataset): cifar dataset loaded into GPU memory
        val_size (int): size of validation set

    Output:
        loader_gpu (dict of DataLoaderGPU): dictionary with 'val', 'train', 'train_small' keys
    """
    
    train_size = len(cifar_gpu[0]) - val_size
    train_small_size = val_size

    loader_train_small_gpu = DataLoaderGPU(cifar_gpu, 
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(np.random.randint(train_size, size=train_small_size)))
    loader_val_gpu = DataLoaderGPU(cifar_gpu,
                                   batch_size=batch_size,
                                   sampler=ChunkSampler(val_size, start=train_size))
    loader_train_gpu = DataLoaderGPU(cifar_gpu,
                                    batch_size=batch_size,
                                    sampler=ChunkSampler(train_size))
    loader_gpu = {
        'val' : loader_val_gpu,
        'train' : loader_train_gpu,
        'train_small' : loader_train_small_gpu
    }
    return loader_gpu


def get_loader(cifar, num_val=1024, batch_size=512):
    num_train = len(cifar) - num_val
    num_train_small = num_val

    loader_train_small = DataLoader(cifar, 
                                batch_size=64, 
                                sampler=SubsetRandomSampler(np.random.randint(num_train, size=num_train_small)))
    loader_val = DataLoader(cifar, 
                            batch_size=64, 
                            sampler=ChunkSampler(num_val, num_train))
    loader_train = DataLoader(cifar,    
                                batch_size=batch_size,
                                sampler=ChunkSampler(num_train))
    loader = {
        'val': loader_val,
        'train': loader_train,
        'train_small': loader_train_small
    }
    return loader


class DataLoaderGPU():
    """ Simple data loader from GPU. 
    Arguments:
        dataset_gpu (dataset): dataset loaded to GPU
        batch_size (int, optional): how many samples per batch to load (default: 1)
        sampler (Sampler): defines strategy to draw samples from dataset
    """    
    def __init__(self, dataset_gpu = None, batch_size = 1, sampler = None):
        self.dataset = dataset_gpu
        self.batch_size = batch_size
        self.sampler = sampler
    
    def __iter__(self):
        self._sampler_iter = iter(self.sampler)
        return self

    def __next__(self):
        index = self._next_index()  # will raise StopIteration
        data = self._next_data(index)
        return data

    def _next_index(self):
        index = []
        for _ in range(self.batch_size):
            index.append(next(self._sampler_iter))
        return torch.tensor(index).type(torch.cuda.LongTensor)  # pylint: disable=not-callable
    
    def _next_data(self, index):
        x = torch.index_select(self.dataset[0], 0, index)
        y = torch.index_select(self.dataset[1], 0, index)
        data = (x, y)
        return data

    def __len__(self):
        return len(self.sampler)


class ChunkSampler(Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples (int): # of desired datapoints
        start (int, optional): offset where we should start selecting from (default: 0)
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


