import sys
sys.path.append("..")

import torch
from calflops import calculate_flops
import sophius.utils as utils
import sqlite3
from sophius.utils import hash_dict

import sophius.dataload as dload
from sophius.modelgen import ConvModelGenerator
from sophius.train import train_express_gpu
import torchvision.datasets as dset
import torchvision.transforms as T
from sophius.encode import Encoder
from sophius.utils import calc_model_flops, hash_dict
import sqlite3
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
from sophius.train import train_on_gpu_ex

def main():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print('Loading CIFAR10 ... ', end='')
    cifar10 = dset.CIFAR10('../data/CIFAR10', train=True, download=True, transform=normalize)

    print('Transfer dataset to gpu ... ', end='')
    cifar_gpu = dload.cifar_to_gpu(cifar10)
    print('Done')

    encoder = Encoder()

    train_params = {
        'val_size': 10000,
        'batch_size': 256,
        'num_epoch': 5,
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

    print('Creating experiment')

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
        else:
            # check if experiment exists
            res = conn.execute('SELECT id FROM experiments WHERE hash == ?', (exp_hash,)).fetchone()
            if res:
                exp_id = res[0]
                print('Experiment exists, exp_id:', exp_id)


            else:
                print('Experiment created, exp_id:', exp_id)
                df.astype(str).to_sql('experiments', conn, if_exists='append')

    # Generate models
    model_gen = ConvModelGenerator((3, 32, 32), 10, conv_num=1, lin_num=1)

    def generator():
        while True:
            yield

    for _ in tqdm(generator()):
        model_tmpl = model_gen.generate_model_tmpl()
        model = model_tmpl.instantiate_model().type(torch.cuda.FloatTensor)

        epoch_results = train_on_gpu_ex(
            model=model,
            dataset=cifar_gpu,
            verbose=False,
            **train_params,
        )

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

        with sqlite3.connect('../data/models.db') as conn:
            df = pd.DataFrame([model_results]).set_index('id').astype(str)
            df.to_sql('models', conn, if_exists='append')

            epoch_results.to_sql('model_epochs', conn, if_exists='append', index=False)


if __name__ == '__main__':
    main()