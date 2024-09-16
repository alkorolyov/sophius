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
    estimator = torch.load('../data/models/estimator_v1.pth').cpu()

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

    val_threshold = 0.6

    def estimate_val_acc(model_tmpl):
        t = torch.tensor(encoder.model2vec(model_tmpl), dtype=torch.float32)
        return estimator.cpu()(t).item()

    print('===> Creating experiment')

    exp_params = {**train_params, **{'in_shape': (3, 32, 32), 'out_shpape': 10}}
    print(exp_params)

    exp_id = 0
    with sqlite3.connect('../data/models.db') as conn:
        try:
            exp_id = conn.execute('SELECT COUNT(*) FROM experiments').fetchone()[0]
        except:
            pass

        df = pd.DataFrame([exp_params], index=[exp_id])
        exp_hash = hash_dict(exp_params)
        df['hash'] = exp_hash
        df.index.name = 'id'
        if exp_id == 0:
            df.astype(str).to_sql('experiments', conn, if_exists='append')
            print('New experiment created, exp_id:', exp_id)
        else:
            # check if experiment exists
            res = conn.execute('SELECT id FROM experiments WHERE hash == ?', (exp_hash,)).fetchone()
            if res:
                exp_id = res[0]
                print('Experiment exists, exp_id:', exp_id)
            else:
                df.astype(str).to_sql('experiments', conn, if_exists='append')
                print('New experiment created, exp_id:', exp_id)

    print('===> Generating models')
    model_gen = ConvModelGenerator((3, 32, 32), 10, conv_num=16, lin_num=3)

    best_model = {'val_acc': 0}
    pb = tqdm()
    while True:
        model_tmpl = model_gen.generate_model_tmpl()
        model = model_tmpl.instantiate_model().type(torch.cuda.FloatTensor)

        # skip below estimated threshold
        if estimate_val_acc(model_tmpl) < val_threshold:
            continue

        epoch_results = train_on_gpu_ex(
            model=model,
            dataset=cifar_gpu,
            verbose=False,
            **train_params,
        )

        # get model_id
        model_id = 0
        with sqlite3.connect('../data/models.db') as conn:
            try:
                model_id = conn.execute('SELECT COUNT(*) FROM models').fetchone()[0]
            except:
                pass

        model_results = calc_model_flops(model, (3, 32, 32))
        model_results['id'] = model_id
        model_results['exp_id'] = exp_id
        model_results['hash'] = encoder.model2hash(model_tmpl)
        model_results['val_acc'] = epoch_results.val_acc.iloc[-10:].mean()
        model_results['train_acc'] = epoch_results.train_acc.iloc[-10:].mean()
        model_results['time'] = epoch_results.time.iloc[-1]

        epoch_results['model_id'] = model_id
        epoch_results['exp_id'] = exp_id

        if model_results['val_acc'] > best_model['val_acc']:
            best_model['val_acc'] = model_results['val_acc']
            best_model['train_acc'] = model_results['train_acc']
            best_model['conv_layers'] = model_tmpl.get_conv_len()

        with sqlite3.connect('../data/models.db') as conn:
            df = pd.DataFrame([model_results]).set_index('id').astype(str)
            df.to_sql('models', conn, if_exists='append')
            epoch_results.to_sql('model_epochs', conn, if_exists='append', index=False)

        pb.update(1)
        desc_msg = (f"best: val {best_model['val_acc']:.3f} | "
                    f"conv {best_model['conv_layers']:2d} | "
                    f"curr: val {model_results['val_acc']:.3f} | "
                    f"train {model_results['train_acc']:.3f} | "
                    f"conv {model_tmpl.get_conv_len():2d} | "
                    f"time {model_results['time'].astype(int):3d}s")
        pb.set_description(desc_msg)

if __name__ == '__main__':
    main()