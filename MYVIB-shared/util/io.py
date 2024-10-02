import os
import numpy as np


def get_file(mode,args):
    file = '/home/zz/wenhaibin/VIB-shared/saved/{}/{}/{}/{}/'.format(mode,args.dataset,args.model,args.flod)
    if not os.path.exists(file):
        os.makedirs(file)
    return file

# noise 参数补上
def get_save_file(args):
    return get_file('save',args)

def get_fig_file(args):
    return get_file('fig',args)

def save_as_numpy(data,name,args):
    np.save(get_save_file(args) + '{}.npy'.format(name),np.array(data)) 