import argparse
import torch
import numpy as np
import random
def get_argparse(TERMINAL = True):
    parser = argparse.ArgumentParser()

    # ------------------------------------dataset------------------------
    parser.add_argument('--dataset', type=str,default='cifar10')
    parser.add_argument('--flod', type=int, default=2)
    parser.add_argument('--tune', action='store_true',default=False)  

  
    parser.add_argument('--steps',type = float,default = 10) 
    parser.add_argument('--eps',type = float,default = 1/255) 
    parser.add_argument('--alpha',type = float,default = 2/255) 
    
    # long tail
    parser.add_argument('--ir',type = float,default = 100) 
    parser.add_argument('--lt_data',action= 'store_true',default=False) 
    parser.add_argument('--lt_test',action= 'store_true',default=False)
    parser.add_argument('--graph_data',action= 'store_true',default=False) 

    #------------------------------------model---------------------------
    parser.add_argument('-U32','--using_resnet32',action= 'store_true',default=False)
    parser.add_argument('--K',type=int ,default= 128)
    parser.add_argument('--model',type=str,default='vib')
    parser.add_argument("--squared", action='store_true',default=False)

    
    #------------------------------------train---------------------------
    parser.add_argument("--seed", type=int,default = 0)
    parser.add_argument('--batch_size', type=int, default=128)   
    parser.add_argument('--save_data',action= 'store_true',default=False) 
    parser.add_argument('-lr','--learning_rate',type = float,default = 1e-3)       
    
    #-----------------------------------attack---------------------------
    parser.add_argument("-act",'--attack', action='store_true',default=True)
    parser.add_argument("-aat",'--Attack_all_types', action='store_true',default=True)
    parser.add_argument('--attack_type',type=str,default='Jitter')
    
    #-----------------------------------loss----------------------------
    parser.add_argument('--kl_eps',type = float,default = 0.01)     

    #-----------------------------------Mback-----------------------------
    parser.add_argument("--stch",action='store_true',default=False)
    parser.add_argument("--stch_mu",type = float,default = 0.2)
    parser.add_argument("--mgda", action='store_true',default=False)
    parser.add_argument("--chs", action='store_true',default=False)
    parser.add_argument("--pcg", action='store_true',default=False)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--mgda_mode", type = str,default='none')#choices=['l2', 'loss', 'loss+','none']
    
    args = parser.parse_args() if TERMINAL else parser.parse_args(args=[])# 给jupyter运行环境留出操作空间
    
    args.cuda = torch.cuda.is_available()
    setup_seed(args.seed)
    # reset
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False