import torchvision
from torchvision import transforms

def get_mnist_dataset(args):

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name =  args.dataset.lower()

    if dataset_name  == 'mnist':
        train_set = torchvision.datasets.MNIST(root = '/home/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.MNIST(root = '/home/data',train=False,download=True,transform=transform)

    elif dataset_name  == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root = '/home/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root = '/home/data',train=False,download=True,transform=transform)
    elif dataset_name == 'kuzu':
        train_set = torchvision.datasets.KMNIST(root = '/home/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.KMNIST(root = '/home/data',train=False,download=True,transform=transform)
    else:
        raise RuntimeError ('only support [mnist ,fashion ,kuzu] ')
    
    return train_set,test_set    
    

