import GPUtil
import os

def available_gpu():
    d = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.7, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    return None if len(d) == 0 else d[0]

def select_gpu():
    d = available_gpu()
    if d is None:
        print('all gpu\'s are busy, please use a CPU or try again later')
    else:
        print(f'using gpu {d}')
        os.environ["CUDA_VISIBLE_DEVICES"]=str(d)
    return ''

list_gpus = GPUtil.showUtilization

if 'GPU' not in os.environ:
    os.environ['GPU'] = select_gpu()
else:
    try:
        if len(os.environ['GPU']) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['GPU']
    except: pass

