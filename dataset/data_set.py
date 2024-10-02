import numpy as np
import torch
from functools import partial

def get_data_set(args):
    return twice_set_for_data(
        partial(first_set_for_data,args),args
    )

def first_set_for_data(args):
    data_name = args.dataset.lower()

    if args.lt_data:
        from .lt.cifar_lt import get_cifar_lt_dataset
        return get_cifar_lt_dataset(args)
    elif args.graph_data:
        raise RuntimeError('has no graph data')
    else:      
        if data_name in ['mnist' ,'fashion','kuzu']:
            from .norm.mnist import get_mnist_dataset
            return get_mnist_dataset(args)
        if 'cifar' in data_name:
            from .norm.cifar import get_cifar_dataset
            return get_cifar_dataset(args)

def twice_set_for_data(first_func,args):
    '''
    do some thing to supplement some attribute on dataset
    tune better off when run cifar set
    '''
    trainset,testset = first_func()
    trainset.targets = np.array(trainset.targets)

    index = trainset.targets.argsort()

    cls_num_index = [(trainset.targets == i).sum() for i in range(len(trainset.classes))]
    for i in range(1,10):
        cls_num_index[i] += cls_num_index[i-1]
    cls_num_index = [0] + cls_num_index
    
    flod_size = int(index.shape[0]/len(trainset.classes)/5)
    flod = args.flod#[0,1,2,3,4]
    flod_data_index =  np.concatenate([index[cls_num_index[j] + flod*flod_size:cls_num_index[j] + (flod + 1)*flod_size]  for j in range(len(trainset.classes))])  

    
    trainset.data = trainset.data[flod_data_index]
    trainset.targets = np.array(trainset.targets[flod_data_index]) 

    trainset.class_counter = torch.tensor(data_class_counter(trainset))
    testset.class_counter = torch.tensor(data_class_counter(testset))

    #  DO some other things
    return trainset,testset

def data_class_counter(dataset):
    if not isinstance(dataset.targets,np.ndarray):
        dataset.targets = np.array(dataset.targets)

    sample_indices = [np.argwhere(dataset.targets == label)[:, 0].tolist() for label in range(len(dataset.classes))]
    class_counter = [len(samples) for samples in sample_indices]
    return class_counter
