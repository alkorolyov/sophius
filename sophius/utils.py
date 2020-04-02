import random
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