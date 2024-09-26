import time

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim

import sophius.utils as utils
from sophius.dataload import get_loader_gpu
import tqdm

import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR


def _instance_from_name(class_name, *args, **kwargs):
    """
    Instantiate class by name
    """
    klass = globals()[class_name]
    class_instance = klass(*args, **kwargs)
    return class_instance


def train_express_gpu(model=None,
                      loader=None,
                      num_epoch=1,
                      milestones=None,
                      train=False,
                      manual_seed=False,
                      verbose=False):
    if milestones is None:
        milestones = []

    start_time = time.time()
    
    if manual_seed:
        torch.cuda.random.manual_seed(12345)
    # init
    loss_fn = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    model.apply(utils.reset)
    model.train()

    train_acc, val_acc = 0, 0

    for i in range(num_epoch):
        running_loss = 0.0
        for t, (x, y) in enumerate(loader['train']):
            scores = model(x)
            loss = loss_fn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        elapsed_time = time.time() - start_time

        if verbose:
            print(f'{i} / {num_epoch}: Loss {running_loss:.4f} {elapsed_time:.0f}s', end='\r')



    
    val_acc = check_accuracy(model, loader['val'])
    if train: 
        train_acc = check_accuracy(model, loader['train_small'])

    time_elapsed = (time.time() - start_time)

    if verbose:
        print ('Finished in %s' % utils.format_time(time_elapsed))
        print('val_acc: %.3f, train_acc: %.3f' % (val_acc, train_acc))
    
    torch.cuda.empty_cache()
    del model
    
    return time_elapsed, val_acc, train_acc

def train_on_gpu(
        model=None,
        loader=None,
        num_epoch=1,
        learning_rate=1e-3,
        gamma=0.1,
        milestones=None,
        random_seed=42,
        verbose=False,
):
    # init

    if milestones is None:
        milestones = []

    start_time = time.time()

    if random_seed:
        utils.set_global_seed(random_seed)

    loss_fn = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    model.apply(utils.reset)
    model.train()

    res = pd.DataFrame(columns=['epoch', 'loss', 'train_acc', 'val_acc', 'time'])

    val_acc, train_acc = 0, 0
    if verbose:
        pb = tqdm.tqdm(total=num_epoch)

    for i in range(num_epoch):
        # model.train()
        running_loss = 0.0
        for t, (x, y) in enumerate(loader['train']):
            scores = model(x)
            loss = loss_fn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        elapsed_time = time.time() - start_time
        val_acc = check_accuracy(model, loader['val'])
        train_acc = check_accuracy(model, loader['train_small'])
        res.loc[i] = [i, running_loss, train_acc, val_acc, elapsed_time]

        if verbose:
            pb.update(1)
            pb.set_description(f'Loss {running_loss:.3f}')

    if verbose:
        print('val_acc: %.3f train_acc: %.3f' % (res.val_acc[-10:].mean(), train_acc))

    torch.cuda.empty_cache()
    del model

    return res


def train_on_gpu_ex(
        model,
        dataset,
        val_size=10000,
        batch_size=1024,
        num_epoch=1,
        optimizer=None,
        opt_params=None,
        scheduler=None,
        sch_params=None,
        random_seed=42,
        verbose=False,
):
    if random_seed:
        utils.set_global_seed(random_seed)

    # init
    loader = get_loader_gpu(dataset, val_size=val_size, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)

    if optimizer:
        optimizer = _instance_from_name(optimizer, model.parameters(), **opt_params)
    else:
        optimizer = optim.AdamW(model.parameters())

    if scheduler:
        scheduler = _instance_from_name(scheduler, optimizer, **sch_params)

    model.apply(utils.reset)
    model.train()

    res = pd.DataFrame(columns=['epoch', 'loss', 'train_acc', 'val_acc', 'time'])

    val_acc, train_acc = 0, 0
    if verbose:
        pb = tqdm.tqdm(total=num_epoch)

    tic = time.time()

    for i in range(num_epoch):
        # model.train()
        running_loss = 0.0
        for t, (x, y) in enumerate(loader['train']):
            scores = model(x)
            loss = loss_fn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()

        toc = time.time()
        val_acc = check_accuracy(model, loader['val'])
        train_acc = check_accuracy(model, loader['train_small'])
        res.loc[i] = [i, running_loss, train_acc, val_acc, toc - tic]

        if verbose:
            pb.update(1)
            pb.set_description(f'Loss {running_loss:.3f}')

    if verbose:
        print('val_acc: %.3f train_acc: %.3f' % (res.val_acc[-10:].mean(), train_acc))

    torch.cuda.empty_cache()
    del model

    return res

def validate_model(model=None, 
                   num_iter=1, 
                   num_epoch=1,
                   train=False,
                   milestones=None,
                   loader=None, 
                   verbose=False):
    val_acc_list = []
    train_acc_list = []
    t_list = []
    if milestones is None:
        milestones = []

    for i in range(num_iter):      
        t, val_acc, train_acc = train_express_gpu(model = model,
                                            loader = loader,
                                            train = train,
                                            milestones = milestones,
                                            num_epoch = num_epoch,
                                            verbose = verbose)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)
        t_list.append(t)
    
    val_mean = np.mean(val_acc_list)
    train_mean = np.mean(train_acc_list)
    val_std = np.std(val_acc_list)
    train_std = np.std(train_acc_list)
    t_mean = np.mean(t_list)
    
    # torch.cuda.empty_cache()
    # del model
    
    if train:
        if verbose:
            print('%d iters: %.1fs \nval: %.3f +- %.3f train: %.3f +- %.3f' % (num_iter, t_mean, val_mean, val_std, train_mean, train_std))
        return t_mean, val_mean, train_mean
    else:
        if verbose:
            print('%d iters: %s \nval: %.3f +- %.3f' % (num_iter, format_time(t_mean), val_mean, val_std))
        return t_mean, val_mean


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        for x, y in loader:
            x_var = x.type(torch.cuda.FloatTensor)
            y_var = y.type(torch.LongTensor)
            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y_var).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
#     print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def check_accuracy_gpu(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, preds = scores.data.cuda().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
#     print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc