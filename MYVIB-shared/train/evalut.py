
import torch
from util.tools import *
import math
import torchattacks
import torch.nn.functional as F
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# Estimating Mutual Information
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def gaussian_log_density_marginal(sample, params, mesh=False):
        """
        Estimate Gaussian log densities:
            For not mesh:
                log p(sample_i|params_i), i in [batch]
            Otherwise:
                log p(sample_i|params_j), i in [num_samples], j in [num_params]
        :param sample: (num_samples, dims)
        :param params: mu, std. Each is (num_params, dims)
        :param mesh:
        :return:
            For not mesh: (num_sample, dims)
            Otherwise: (num_sample, num_params, dims)
        """
        # Get data
        mu, std = params
        # Mesh
        if mesh:
            sample = sample.unsqueeze(1)
            mu, std = mu.unsqueeze(0), std.unsqueeze(0)
        # Calculate
        # (1) log(2*pi)
        constant = math.log(2 * torch.pi)
        # (2) 2 * log std_i
        log_det_std = 2 * torch.log(std)
        # (3) (x-mu)_i^2 / std_i^2
        dev_inv_std_dev = ((sample - mu) / std) ** 2
        # Get result
        log_prob_marginal = - 0.5 * (constant + log_det_std + dev_inv_std_dev)
        # Return
        return log_prob_marginal
@torch.no_grad()
def eval_mi_x_z_monte_carlo(batch_z_data, batch_x_data_mu,batch_x_data_std,len_x_data):
    '''
    batch_x_data  :  ib_out ['mu', 'std'].
    batch_z : ib_out['x']
    
    '''
    ent_x_z = []
    # Eval H(X|Z)
    for batch_z_index,batch_z in enumerate(batch_z_data):
        # Get z. (batch, nz)
        # Calculate H(X|batch_z)
        # 1. Get log p(batch_z|x). (batch, total_num_x)
        log_p_batch_z_x = []
        for batch_x_index, batch_x in enumerate(zip(batch_x_data_mu,batch_x_data_std)):
            show_progress(
                "Estimating I(X;Z)", index=batch_z_index * len(batch_x_data_mu) + batch_x_index,
                maximum=len(batch_x_data_mu) * len(batch_x_data_mu))
            # (1) Get params (mu, std). (batch, nz)
            # (2) Get log p(batch_z|batch_x). (batch, batch)
            log_p_batch_z_batch_x = gaussian_log_density_marginal(batch_z, batch_x, mesh=True).sum(dim=2)
            # Accumulate
            log_p_batch_z_x.append(log_p_batch_z_batch_x)  

        log_p_batch_z_x = torch.cat(log_p_batch_z_x, dim=1)
        # 2. Normalize to get log p(x|batch_z). (batch, total_num_x)
        log_p_x_batch_z = log_p_batch_z_x - torch.logsumexp(log_p_batch_z_x, dim=1, keepdim=True)
        # 3. Get H(X|batch_z). (batch, )
        ent_x_batch_z = (-torch.exp(log_p_x_batch_z) * log_p_x_batch_z).sum(dim=1)
        # Accumulate
        ent_x_z.append(ent_x_batch_z)
    ent_x_z = torch.cat(ent_x_z, dim=0).mean()
    ################################################################################################################
    # Eval H(X)
    ################################################################################################################
    ent_x = math.log(len_x_data)
    ################################################################################################################
    # Eval I(X;Z) = H(X) - H(X|Z)
    ################################################################################################################
    ret = ent_x - ent_x_z
    # Return
    return ret
@torch.no_grad()
def eval_mi_y_z_variational_lb(prob_data,class_counter):
    ####################################################################################################################
    # Eval H(Y|Z) upper bound.
    ####################################################################################################################
    ent_y_z = []
    for batch_index, prob in enumerate(prob_data):
        show_progress("Estimating I(Z;Y)", batch_index, len(prob_data))
        prob = torch.softmax(prob, dim=1)
        ent_y_batch_z = (-prob * torch.log(prob + 1e-10)).sum(dim=1)
        # Accumulate to result
        ent_y_z.append(ent_y_batch_z)
    
    ent_y_z = torch.cat(ent_y_z, dim=0).mean()
    ####################################################################################################################
    # Get H(Y)
    ####################################################################################################################
    
    class_prob = class_counter / class_counter.sum()
    ent_y = (-class_prob * torch.log(class_prob)).sum()
    ####################################################################################################################
    # Eval I(Y;Z) = H(Y) - H(Y|Z)
    ####################################################################################################################
    ret = ent_y - ent_y_z
    # Return
    return ret

    
class Attack_evalut:
    def __init__(self,model,num_classes,args):
        # 为了应对torchattack库只让模型有一个输出的问题 需要修改torchattacks的源代码
        # was changeing file  torchattacks.attack.get_logits()
        eps,alpha,steps = args.eps,args.alpha,args.steps
        if 'cifar' in args.dataset:
            eps,alpha,steps = 1/255,2/255,10
        else:
            eps,alpha,steps = 8/255,10/255,10

        self.attack_types =  {
            'FGSM':torchattacks.FGSM(model, eps=eps),
            'PGD':torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True),
            'NIFGSM':torchattacks.NIFGSM(model, eps = eps, alpha=alpha, steps = steps),
            'EOTPGD':torchattacks.EOTPGD(model, eps=eps, alpha=alpha, steps=steps, eot_iter=2),
            'MIFGSM':torchattacks.MIFGSM(model, eps=eps, steps=steps, decay=1.0),
            'UPGD':torchattacks.UPGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True),
            'Jitter':torchattacks.Jitter(model, eps=eps, alpha=alpha, steps=steps, scale=10, std=0.1, random_start=True)
        }
        self.attack = self.attack_types[args.attack_type]
    
    def Attack_epoch(self,model,test_loder,args,desc = 'AttackTest'):
        model.eval()
        sum_acc = 0.
        info_losses,ce_losses = [],[] 
        
        for data,label in tqdm(test_loder,desc=desc):
            if args.cuda:
                data = data.cuda()
                label = label.cuda()        
            
            atc_data = self.attack(data, label)
            with torch.no_grad():
                out,ib_out = model(atc_data)
                     
                info_loss = model.get_info_loss(ib_out,squared = args.squared) 
                ce_loss = F.nll_loss(out,label)
                
                sum_acc += (torch.argmax(out,dim=1) == label).sum()
                info_losses.append(info_loss)
                ce_losses.append(ce_loss)


        item = test_loder.dataset.data.shape[0]

        Attack_info = {
            'info_loss':(sum(info_losses)/len(info_losses)).item(),
            'ce_loss':(sum(ce_losses)/len(ce_losses)).item(),
            'acc':(sum_acc/item).item()
        }

        return Attack_info
    
    def Attack_epoch_all_types(self,model,test_loder,args):
        ret = {}
        for name,adv in self.attack_types.items():
            self.attack = adv
            ret[name] = self.Attack_epoch(model,test_loder,args,'{} : AttackTest'.format(name))
            torch.cuda.empty_cache()
        
        return ret

    


