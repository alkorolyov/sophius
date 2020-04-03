import random, gc, bitmath, torch
from functools import reduce
import operator as op

def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def shuffle_dict(d):
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    return d

def print_nonprivate_properties(obj):
    print('%-20s %s' % ('name', obj.__class__.__name__))
    for k, v in vars(obj).items():
        if not k.startswith('_'):
            print('%-20s %-5s %s' % (k, type(v), v))
    return

def print_properties(obj):
    print('%-20s %s' % ('name', obj.__class__.__name__))
    for k, v in vars(obj).items():
        print('%-20s %-5s %s' % (k, type(v), v))
    return

def format_time(sec):
    time_str = ''
    days = int(sec // 86400)
    sec = sec % 86400
    hours = int(sec // 3600)
    sec = sec % 3600
    minutes = int(sec // 60)
    sec = sec % 60

    time_str = ''

    if days > 0:
        time_str += '{:d}d {:d}h {:d}m '.format(days, hours, minutes)
    elif hours >0:
        time_str += '{:d}h {:d}m'.format(hours, minutes)
    elif minutes > 0:
        time_str = '{:d}m '.format(minutes)
    time_str += '{:.1f}s '.format(sec)
    
    return time_str

def format_number(num):
    if num // 10 ** 6 > 0:
        return str(round(num / 10 ** 6, 2)) + 'M'
    elif num // 10 ** 3:
        return str(round(num / 10 ** 3, 2)) + 'k'
    else:
        return str(num)

def get_tensors_memory():
    tensors_list = []
    for obj in gc.get_objects():        
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_size = reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0
                mem_size = bitmath.Byte(tensor_size * obj.element_size() )
                tensor_size_formated = mem_size.best_prefix().format("{value:>6.2f} {unit:5}")
                tensors_list.append({'size' : mem_size,
                                     'shape' : str(list(obj.size())),
                                     'type' : str(type(obj)),
                                     'dtype' : str(obj.dtype),
                                     'data_size' : obj.element_size(),
                                     'device' : str(obj.device),
                                     'class' : obj.__class__.__name__,
                                    })
        except:
            pass
    for tensor in sorted(tensors_list, key=lambda d: (d['class'], d['size']), reverse=True):
        tensor_size_formated = tensor['size'].best_prefix().format("{value:>6.2f} {unit:5}")
        tensor_info_formated = '{p[shape]:20} {p[dtype]:16} {p[device]:8} {p[class]:10}'.format(p=tensor)
        print(tensor_size_formated, tensor_info_formated)
