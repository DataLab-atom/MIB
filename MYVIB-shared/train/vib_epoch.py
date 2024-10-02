from tqdm import tqdm
import torch
import torch.nn.functional as F
from .evalut import *
from util.tools import *

def train_epoch(model,scheduler,train_loder,mback,args):
    model.train()
    info_losses,ce_losses = [],[] 
    sum_acc = 0.
    item = 0
    mback_mgda_weight = []
    for data,label in tqdm(train_loder,desc='Train'):
        if args.cuda:
            data = data.cuda()
            label = label.cuda()        
        
        out,ib_out = model(data)
        info_loss = model.get_info_loss(ib_out,squared = args.squared)
        ce_loss = F.nll_loss(out,label)
        mback.backward([info_loss,ce_loss])
        #mback_mgda_weight.append(mback.mgda_sol)
        info_losses.append(info_loss)
        ce_losses.append(ce_loss)

        sum_acc += (torch.argmax(out,dim=1) == label).sum()
        item += data.shape[0]

    scheduler.step()
    
    train_epoch_info = {
        'info_loss':(sum(info_losses)/len(info_losses)).item(),
        'ce_loss':(sum(ce_losses)/len(ce_losses)).item(),
        'acc':(sum_acc/item).item(),
        #'mgda':sum(mback_mgda_weight)/len(mback_mgda_weight)
    }

    return train_epoch_info

@torch.no_grad()
def eval_epoch(model,test_loder,args):
    model.eval()

    info_losses,ce_losses = [],[] 
    sum_acc = 0.
    
    for data,label in tqdm(test_loder,desc='Test'):
        if args.cuda:
            data = data.cuda()
            label = label.cuda()        
        
        out,ib_out = model(data)
        
        info_loss = model.get_info_loss(ib_out,squared = args.squared)
        ce_loss = F.nll_loss(out,label)

        info_losses.append(info_loss)
        ce_losses.append(ce_loss)
        sum_acc += (torch.argmax(out,dim=1) == label).sum()

    item = test_loder.dataset.data.shape[0]
    test_epoch_info = {
        'info_loss':(sum(info_losses)/len(info_losses)).item(),
        'ce_loss':(sum(ce_losses)/len(ce_losses)).item(),
        'acc':(sum_acc/item).item()
    }

    return test_epoch_info

@torch.no_grad()
def eval_mi(model,loder,args):
    
    prob_data,batch_z_data, batch_x_data_mu,batch_x_data_std = [],[],[],[]
    
    for data,label in tqdm(loder,desc='Preing'):
        if args.cuda:
            data = data.cuda()
            label = label.cuda()        
        
        out,ib_out = model(data)
        
        prob_data.append(out)
        batch_z_data.append(ib_out['x'])
        batch_x_data_mu.append(ib_out['mu'])
        batch_x_data_std.append(ib_out['std'])

    mix_z =  eval_mi_x_z_monte_carlo(batch_z_data, batch_x_data_mu,batch_x_data_std,loder.dataset.data.shape[0])
    miy_z = eval_mi_y_z_variational_lb(prob_data,loder.dataset.class_counter)
    return mix_z,miy_z