# sophius
 Model and Layer templates Generator for pytroch

 Works as a Keras analogue. Model Generator can generate model templates with random number
 of convolutional and linear layers, with optional auxialary layers between them. Automatically
 fixes input and output shapes sizes for all torch Modules. 

 # Example usage

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
<the actual output here>
Conv2d       (16, 8, 8)     (1, 1)   (4, 4)  
LeakyReLU    (16, 8, 8)     (0.001) 
BatchNorm2d  (16, 8, 8)    
Flatten      1024          
Linear       1024          
ReLU         1024          
Linear       10

# Instantiate model to torch model
fixed_model_gpu = model_tmpl.instantiate_model().type(torch.cuda.FloatTensor)
print(fixed_model_gpu)

Sequential(
  (0): Conv2d(3, 16, kernel_size=(1, 1), stride=(4, 4))
  (1): LeakyReLU(negative_slope=0.001)
  (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Flatten()
  (4): Linear(in_features=1024, out_features=1024, bias=True)
  (5): ReLU()
  (6): Linear(in_features=1024, out_features=10, bias=True)
)

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
Finished in 8.3s 
val_acc: 0.468, train_acc: 0.462
```


```
python -m ipykernel install --user --name sophius
```

 # To install locally for edit
 pip install -e <local sophius folder>
 Example: 
 pip install -e c:\Users\Alex\Documents\Anaconda3\projects\sophius

# edit pytest.ini
Edit .vscode/settings.json and for pytest support:
    "python.testing.pytestArgs": [
        "-o", "junit_family=xunit1"
    ],
