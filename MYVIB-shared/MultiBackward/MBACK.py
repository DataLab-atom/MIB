from .LossFun.mgda.min_norm_solvers import MinNormSolver
from .LossFun.ChebShev import Chebyshev
from .GradFun.pcgrad import PCGrad
import time
from torch.autograd import Variable
from functools import partial
import torch

class MBACK():
    def __init__(self,
                 optimizer,
                 args,
                 mgda_encoder = None):
        
        self.optimizer = optimizer
        self.args = args

        if args.mgda:
            assert mgda_encoder != None
            self.mgda_encoder = mgda_encoder
        if args.pcg:
            self.pcg_opt = PCGrad(optimizer)
        elif args.chs:
            self.chebshev = Chebyshev()
    
    def backward(self,losses):
        if self.args.mgda:
            losses = self.mgda(losses,task = self.args.tasks, gn_mode = self.args.mgda_mode)
        
        if self.args.pcg:
            return self.pcg(losses)
        elif self.args.chs:
            return self.chs(losses)
        elif self.args.stch:
            return self.stch(losses)
        
        return self.base(losses)
    


    def mgda(self,losses,task=None,gn_mode = 'none'):
        def get_parameters_grad(model):
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(Variable(param.grad.data.clone(), requires_grad=False))

            return grads
        
        loss_data = {}
        grads = {}
        
        if task==None:
            task = [i for i in range(len(losses))]
        
        for t in task:
            loss = losses[t]
            self.optimizer.zero_grad()
            loss_data[t] = loss.data
            loss.backward(retain_graph=True)
            grads[t] = get_parameters_grad(self.mgda_encoder)    

        gn = MinNormSolver.gradient_normalizers(grads, loss_data, gn_mode)
        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in task],True)
        for i in task:
            losses[i] *= float(sol[i])
        self.mgda_sol = sol
        return losses

    def pcg(self,losses):
        self.pcg_opt.zero_grad()
        self.pcg_opt.pc_backward(losses)
        self.pcg_opt.step()

    def chs(self,losses):
        self.chebshev.append(losses)
        self.optimizer.zero_grad()
        self.chebshev.backward()
        self.optimizer.step()
    
    def stch(self,losses):
        mu = self.args.stch_mu
        self.optimizer.zero_grad()
        loss = torch.logsumexp(torch.stack(losses,dim = 0)/mu,dim = 0)*mu
        loss.backward()
        self.optimizer.step()

    def base(self,losses):
        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    


    