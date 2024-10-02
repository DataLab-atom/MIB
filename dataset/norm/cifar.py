import torchvision
from torchvision import transforms

def get_cifar_dataset(args):

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

    if dataset_name  == 'cifar10':
        train_data = torchvision.datasets.CIFAR10('/home/data', train=True, 
            download=True, transform=transform_train)
        test_data  = torchvision.datasets.CIFAR10('/home/data', train=False, 
            download=True, transform=transform_test) 
    elif dataset_name  == 'cifar100':
        train_data = torchvision.datasets.CIFAR100('/home/data', train=True, 
            download=True, transform=transform_train)
        test_data  = torchvision.datasets.CIFAR100('/home/data', train=False, 
            download=True, transform=transform_test) 
    else:
        raise RuntimeError('only support [cifar10 ,cifar100] ')
    
    return train_data,test_data
