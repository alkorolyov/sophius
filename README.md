# sophius
 Multistep project for automatical neural network architecture search (NAS) for visual recognition tasks (CIFAR10 as example)

 
 # Step 1
 Generate random a set of random models, evaluate their performance and store results in local database.

 For that custom Module Templates and Model Generator was written.
 Model Generator can generate model templates with random number
 of convolutional and linear layers, with optional auxialary layers between them. Automatically
 fixes input and output shapes sizes for all torch Modules. 


```python
import torch
import sophius.utils as utils
import sophius.dataload as dload
from sophius.modelgen import ConvModelGenerator
from sophius.train import train_express_gpu
import torchvision.datasets as dset
import torchvision.transforms as T

# Generate Model Template
model_gen = ConvModelGenerator((3, 32, 32), 10, conv_num=6, lin_num=3)
model_tmpl = model_gen.generate_model_tmpl()
print(model_tmpl)

# Conv2d       (16, 8, 8)     (1, 1)   (4, 4)  
# LeakyReLU    (16, 8, 8)     (0.001) 
# BatchNorm2d  (16, 8, 8)    
# Flatten      1024          
# Linear       1024          
# ReLU         1024          
# Linear       10

# Instantiate model to torch model
fixed_model_gpu = model_tmpl.instantiate_model().cuda()
print(fixed_model_gpu)

# Sequential(
#   (0): Conv2d(3, 16, kernel_size=(1, 1), stride=(4, 4))
#   (1): LeakyReLU(negative_slope=0.001)
#   (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (3): Flatten()
#   (4): Linear(in_features=1024, out_features=1024, bias=True)
#   (5): ReLU()
#   (6): Linear(in_features=1024, out_features=10, bias=True)
# )

# Load CIFAR to gpu memory, using custom dataloader
VAL_SIZE = 1024
cifar10 = dset.CIFAR10('../data/CIFAR10', train=True, download=True,
                           transform=T.ToTensor())
cifar_gpu = dload.cifar_to_gpu(cifar10)
loader_gpu = dload.get_loader_gpu(cifar_gpu, val_size=VAL_SIZE, batch_size=1024)

# Train
t, val_acc, train_acc = train_express_gpu(model = fixed_model_gpu,
                                          train = True,
                                          loader = loader_gpu,
                                          milestones = [],
                                          num_epoch = 1,
                                          verbose = True)
# Finished in 8.3s 
# val_acc: 0.468, train_acc: 0.462
```

# Step 2
Train custom LSTM model to predict validation accuracy of the generated models.

For that each layer was encoded as a 32bit vector and whole architecture is represented as sequence of vectors. Then it is used as an input to custom LSTMRegressor. Which then used to filter only high accuracy models during random generation. After around 2000 randomly generated models we have a good R^2 > 0.8 correlation on the validation set.

Example of bit vector representation of the model

```python
from sophius.templates import Conv2dTmpl
from sophius.encode import Encoder

encoder = Encoder()
t = Conv2dTmpl(out_channels=32, kernel_size=(3, 3), stride=(1, 1))

print(encoder.encode_template(t))
# [0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0]

model_gen = ConvModelGenerator((3, 32, 32), 10, conv_num=1, lin_num=1)
model_tmpl = model_gen.generate_model_tmpl()

print(model_tmpl)
# Conv2d       (8, 10, 10)    (5, 5)   (3, 3)
# PReLU        (8, 10, 10)
# Flatten      800
# Linear       10

print(encoder.model2vec(model_tmpl))
# [0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1]
# [0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]            

```

LSTMRegressor. Dropout 0.5 was necessary to avoid overfitting on small dataset

```python
import torch

class LSTMRegressor(torch.nn.Module):
    def __init__(self,
                 input_dim=32,
                 hidden_dim=32,
                 num_layers=2,
                 dropout=0.5,
                 **_):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
```
Training results on validation set. 

![image](https://github.com/user-attachments/assets/c86d287b-516d-4091-a499-c7dad7653167)

# Step 3
Would be to implement genetic algorithm to further optimize and select the best sequences to reach more than
0.9 accuracy on the validation set.
