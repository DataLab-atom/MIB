import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch

def get_img_num_per_cls(dataset, cls_num, imb_type, imb_ratio):
    img_max = len(dataset.data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_ratio**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_ratio))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(dataset, img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(dataset.targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    dataset.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        dataset.num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(dataset.data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return new_data,torch.tensor(new_targets,dtype= torch.long)
        
def get_cls_num_list(dataset):
    cls_num_list = []
    for i in range(dataset.cls_num):
        cls_num_list.append(dataset.num_per_cls_dict[i])
    return cls_num_list


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', 
                 imb_ratio=100, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        
        img_num_list = get_img_num_per_cls(self,self.cls_num, imb_type, 1.0/imb_ratio)
        self.data,self.targets = gen_imbalanced_data(self,img_num_list)


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', 
                 imb_ratio=100, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        
        img_num_list = get_img_num_per_cls(self,self.cls_num, imb_type, 1.0/imb_ratio)
        self.data,self.targets = gen_imbalanced_data(self,img_num_list)

def get_cifar_lt_dataset(args):

    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
    ])

    dataset_name =  args.dataset.lower()
    ir = args.ir
    lt_test = args.lt_test
    if dataset_name  == 'cifar10':
        train_data = IMBALANCECIFAR10('/home/data',imb_ratio = ir, train=True, 
            download=True, transform=transform_train)
        
        if lt_test:
            test_data  = IMBALANCECIFAR10('/home/data',imb_ratio = ir, train=False, 
                download=True, transform=transform_test) 
        else :
            test_data  = torchvision.datasets.CIFAR10('/home/data', train=False, 
                download=True, transform=transform_test) 
    elif dataset_name  == 'cifar100':
        train_data = IMBALANCECIFAR100('/home/data',imb_ratio = ir, train=True, 
            download=True, transform=transform_train)
        
        if lt_test:
            test_data  = IMBALANCECIFAR100('/home/data',imb_ratio = ir, train=False, 
                download=True, transform=transform_test)
        else:
            test_data  = torchvision.datasets.CIFAR100('/home/data', train=False, 
                download=True, transform=transform_test) 
         
    else:
        raise RuntimeError('only support [cifar10 ,cifar100] ')

    return train_data,test_data
