import GPUtil
import os
import torch

def available_gpu():
    try:
        d = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.7, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        return None if len(d) == 0 else d[0]
    except:
        return -1

def select_gpu():
    d = available_gpu()
    if d is None:
        print('all gpu\'s are busy, please use a CPU or try again later')
        return ''
    elif d >= 0:
        print(f'using gpu {d}')
        os.environ["CUDA_VISIBLE_DEVICES"]=str(d)
        return str(d)
    return ''

list_gpus = GPUtil.showUtilization
torch.set_num_threads(4)

if 'GPU' not in os.environ:
    os.environ['GPU'] = select_gpu()
else:
    try:
        if len(str(os.environ['GPU'])) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['GPU']
    except: pass

