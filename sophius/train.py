import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import sophius.utils as utils

def train_express_gpu(model = None,
                      loader = None,
                      num_epoch = 1,
                      milestones = [],
                      train = False,
                      manual_seed = False,
                      verbose = False):
    
    start_time = time.time()
    
    if manual_seed: 
        torch.cuda.random.manual_seed(12345)
    # init
    loss_fn = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = 0.1)
    model.apply(utils.reset)
    model.train()

    train_acc, val_acc = 0, 0

    for i in range(num_epoch):
        for t, (x, y) in enumerate(loader['train']):
            scores = model(x)
            loss = loss_fn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
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

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
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
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, preds = scores.data.cuda().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
#     print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc