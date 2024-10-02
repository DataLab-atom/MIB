from util.optm import get_argparse
from dataset import get_data_set
from model.VIB import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from MultiBackward import MBACK
from train.vib_epoch import train_epoch,eval_epoch,eval_mi
from train.evalut import Attack_evalut
from util.io import get_file
import numpy as np
import torch
import copy
import numpy as np


from functools import partial
import os 

def get_model(example_input,args,num_class):
    if args.model == 'resnet':
        args.kl_eps = 0.
        return VIB(example_input,args,num_class)
    elif args.model == 'vib':
        return VIB(example_input,args,num_class)
    elif args.model == 'nib':
        return NIB(example_input,args,num_class)
    else :
        raise RuntimeError('model not support main.py line 28')



def main():
    args=get_argparse() 


    args.save_as = 'beta_{}squared_{}'.format(args.kl_eps,int(args.squared))
    args.epochs = 100 if 'cifar' in args.dataset else 40
    if args.squared:
        args.kl_eps *= 0.01
    
    train_set,test_set = get_data_set(args)
    example_input = train_set[0][0].unsqueeze(0)  # 构造一个示例输入
    num_class=len(test_set.classes)
    model =  get_model(example_input,args,num_class)
    
    train_loder = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,pin_memory=True)
    test_loder = DataLoader(test_set,batch_size=args.batch_size*2,shuffle=False,pin_memory=True,num_workers=2)
    
    optimizer = Adam(model.parameters(),lr= args.learning_rate,betas=(0.5, 0.999))
    mback = MBACK(optimizer,args,model.ENC)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** (epoch // 2))

    #######################################################################################################################
    if args.cuda:
        model.cuda()

    Atc_evalut = None
    max_acc = {'mean_accuracy':0.}
    if args.attack:
        Atc_evalut = Attack_evalut(model,num_class,args)

        if args.Attack_all_types:
            for name in Atc_evalut.attack_types.keys():
                max_acc[name] = 0.
        else:
            max_acc[args.attack_type] = 0.

    #######################################################################################################################
    info = {
         'epoch':[]
    }    
    for i in range(args.epochs):#
        epoch_info = {}
        epoch_info['train'] = train_epoch(model,scheduler,train_loder,mback,args)
        epoch_info['test'] = eval_epoch(model,test_loder,args)
        torch.cuda.empty_cache()

        if not Atc_evalut == None:
            if args.Attack_all_types:
                epoch_info['Atc_test'] = Atc_evalut.Attack_epoch_all_types(model,test_loder,args)
            else :
                epoch_info['Atc_test'] = {args.attack_type:Atc_evalut.Attack_epoch(model,test_loder,args)}
        #############################################################################################################################################
        
        
        info['epoch'].append(epoch_info)
        if epoch_info['test']['acc'] > max_acc['mean_accuracy']:
            max_acc['mean_accuracy'] = epoch_info['test']['acc']
        # for name,item in epoch_info['Atc_test'].items():
        #     if item['acc'] > max_acc[name]:
        #         max_acc[name] = item['acc']
        
        print('epoch:[{} / {}]'.format(i,args.epochs))
        print('ACC : ')
        for name,val in max_acc.items():
            print('   --> {} : {:.2f} '.format(name,val*100))          

    '''
    info['mi'] = {}
    for d_name, dataloader in zip(['train', 'test'], [train_loder,test_loder]):
            mix,miy = eval_mi(model,dataloader,args)
            info['mi']['%s_mi_x_class' % d_name],info['mi']['%s_mi_y_class' % d_name] = mix.item(),miy.item()
    print(info['mi'])
    '''
    
    
    np.save(get_file('info',args) + '{}.npy'.format(args.save_as),info)
    torch.save(model.cpu(),get_file('model',args) + '{}.pt'.format(args.save_as))

    return max_acc





if __name__ == '__main__':
  
   main()


   
    