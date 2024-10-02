import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tools import gaussian_kl_div
from .resnet import resnet18_en,resnet32_en


def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        if layer.bias is not None: layer.bias.data.zero_()

class Res_MAY_VIB(nn.Module):
    def __init__(self,
                 example_input,
                 args,
                 num_class = 100):
        super(Res_MAY_VIB,self).__init__()
        
        K = args.K
        self.beta = args.kl_eps
        self.ENC,ENC_out_channels = get_rep(example_input,args)
        self.linler1 = nn.Linear(64,32,bias=False)
        self.linler2 = nn.Linear(32,10,bias=False)
        self.DEC = nn.Sequential(
            self.linler1,
            nn.ReLU(),
            nn.BatchNorm1d(32),
            self.linler2,
            nn.LogSoftmax()
        )

    def forward(self,x):
        ib_out = {}

        x = self.ENC(x)
        x = x + 1e-6
        mu,std = torch.split(x,64,dim=1) 
        eps = torch.randn_like(mu)
        
        #std = F.softplus(std,beta = 1) 
        x = std*eps + mu
         
        ib_out['x']  =  x
        ib_out['mu'] = mu
        ib_out['std'] = std
        
        return self.DEC(x),ib_out

def get_rep(example_input,args):
    if args.using_resnet32:
        return resnet32_en(example_input,args)
    else:
        return resnet18_en(example_input,args)

class DEC(nn.Module):
    def __init__(self,
                 in_channels,
                 num_class = 100):
        super(DEC,self).__init__()
        self.linler = nn.Linear(in_channels,num_class,bias=False)

    def forward(self,x):
        return F.log_softmax(self.linler(x),dim=1)

class VIB(nn.Module):
    def __init__(self,
                 example_input,
                 args,
                 num_class = 100):
        super(VIB,self).__init__()
        
        K = args.K
        self.beta = args.kl_eps
        self.ENC,ENC_out_channels = get_rep(example_input,args)
        self.bn = nn.BatchNorm1d(ENC_out_channels)
        self.linler_mu = nn.Linear(ENC_out_channels,K,bias=False)
        self.linler_std = nn.Linear(ENC_out_channels,K,bias=False)
        self.DEC = DEC(K,num_class)
        self.apply(init_weights)

    def forward(self,x):
        ib_out = {}

        x = self.ENC(x)
        x = self.bn(x)
        mu = F.relu(self.linler_mu(x)) + 1e-6
        std = F.relu(self.linler_std(x)) + 1e-6
        eps = torch.randn_like(mu)
        
        #std = F.softplus(std,beta = 1) 
        x = std*eps + mu
         
        ib_out['x']  =  x
        ib_out['mu'] = mu
        ib_out['std'] = std
        
        return self.DEC(x),ib_out

    def get_info_loss(self,ib_out,squared = False):
        mu,std = ib_out['mu'],ib_out['std']
        loss = self.beta*gaussian_kl_div((mu,std), average_batch=True)
        return loss**2 if squared else loss

class NIB(nn.Module):
    def __init__(self,
                 example_input,
                 args,
                 num_class = 100,
                 log_std = -1,
                 log_std_trainable = True):
        super(NIB,self).__init__()
        
        K = args.K
        self.beta = args.kl_eps
        self.ENC,ENC_out_channels = get_rep(example_input,args)
        self.bn = nn.BatchNorm1d(ENC_out_channels)
        self.linler_mu = nn.Linear(ENC_out_channels,K,bias=False)

        if log_std_trainable:  # 初始化对数方差
            self.register_parameter('log_std', torch.nn.Parameter(torch.FloatTensor([log_std]), requires_grad=True))
        else:
            self.register_buffer('log_std', torch.FloatTensor([log_std]))
        self.DEC = DEC(K,num_class)
        self.apply(init_weights)

    def forward(self,x):
        ib_out = {}

        x = self.ENC(x)
        x = self.bn(x)
        mu = F.relu(self.linler_mu(x)) + 1e-6
        std = self.log_std.exp().expand(*mu.size())
        eps = torch.randn_like(mu)
        
        #std = F.softplus(std,beta = 1) 
        x = std*eps + mu
         
        ib_out['x']  =  x
        ib_out['mu'] = mu
        ib_out['std'] = std
        
        return self.DEC(x),ib_out

    def get_info_loss(self,ib_out,squared = False):
        
        mu,std = ib_out['mu'],ib_out['std']
        loss = self.beta*gaussian_kl_div(params1=(mu.unsqueeze(1), std.unsqueeze(1)),
                                   params2=(mu.unsqueeze(0), std.unsqueeze(0))).mean()
        
        return loss**2 if squared else loss
    



    