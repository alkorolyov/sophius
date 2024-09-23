import sys
sys.path.append("..")

import sqlite3
import pandas as pd
from tqdm import tqdm

import sophius.dataload as dload
from sophius.modelgen import ConvModelGenerator
from sophius.encode import Encoder
from sophius.utils import calc_model_flops, hash_dict
from sophius.train import train_on_gpu_ex
from sophius.estimate import LSTMRegressor
from sophius.db import database, Experiments, Models, ModelEpochs

import torch
import torchvision.datasets as dset
import torchvision.transforms as T


def main():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print('===> Loading CIFAR10 ... ', end='')
    cifar10 = dset.CIFAR10('../data/CIFAR10', train=True, download=True, transform=normalize)

    print('===> Preprocessing dataset ... ', end='')
    cifar_gpu = dload.cifar_to_gpu(cifar10)
    print('Done')


    encoder = Encoder()
    estimator = torch.load('../data/models/estimator_v2.pth').cpu()

    def estimate_val_acc(model_tmpl):
        t = torch.tensor(encoder.model2vec(model_tmpl), dtype=torch.float32)
        return estimator.cpu()(t).item()

    train_params = {
        'val_size': 10000,
        'batch_size': 256,
        'num_epoch': 50,
        'random_seed': 42,
        'optimizer': 'AdamW',
        'opt_params': {
            'lr': 1e-3,
        },
        'scheduler': 'ExponentialLR',
        'sch_params': {
            'gamma': 0.95,
        },
    }

    val_threshold = 0.70

    with database:
        database.create_tables([Experiments, Models, ModelEpochs])

    print('===> Creating experiment')
    exp_params = {**train_params, **{'in_shape': (3, 32, 32), 'out_shape': 10}}
    exp, _ = Experiments.get_or_create(**exp_params)

    print('===> Generating models')
    model_gen = ConvModelGenerator(
        exp_params['in_shape'],
        exp_params['out_shape'],
        conv_num=16, lin_num=3
    )
    best_model = {'val_acc': 0}
    pb = tqdm()

    while True:
        model_tmpl = model_gen.generate_model_tmpl()
        model_gpu = model_tmpl.instantiate_model().cuda()

        # skip below estimated threshold
        if estimate_val_acc(model_tmpl) < val_threshold:
            continue

        epoch_results = train_on_gpu_ex(
            model=model_gpu,
            dataset=cifar_gpu,
            verbose=False,
            **train_params,
        )

        model_info = calc_model_flops(model_gpu, model_gen.in_shape)
        model = Models.create(
            exp_id=exp.id,
            hash=encoder.model2hash(model_tmpl),
            flops=model_info['flops'],
            macs=model_info['macs'],
            params=model_info['params'],
            val_acc=round(epoch_results.val_acc.iloc[-10:].mean(), 4),
            train_acc=round(epoch_results.train_acc.iloc[-10:].mean(), 4),
            time=round(epoch_results.time.iloc[-1], 1),
        )

        for _, row in epoch_results.iterrows():
            ModelEpochs.create(
                exp_id=exp.id,
                model_id=model.id,
                epoch=row['epoch'],
                loss=round(row['loss'], 2),
                train_acc=round(row['train_acc'], 4),
                val_acc=round(row['val_acc'], 4),
                time=round(row['time'], 1),
            )

        if model.val_acc > best_model['val_acc']:
            best_model['val_acc'] = model.val_acc
            best_model['train_acc'] = model.train_acc
            best_model['conv_layers'] = model_tmpl.get_conv_len()

        pb.update(1)
        desc_msg = (f"best: val {best_model['val_acc']:.3f} | "
                    f"conv {best_model['conv_layers']:2d} | "
                    f"curr: val {model.val_acc:.3f} | "
                    f"train {model.train_acc:.3f} | "
                    f"conv {model_tmpl.get_conv_len():2d} | "
                    f"time {model.time.astype(int):3d}s")
        pb.set_description(desc_msg)

if __name__ == '__main__':
    main()