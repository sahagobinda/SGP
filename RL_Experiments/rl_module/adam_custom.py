import numpy as np
import torch
from collections import OrderedDict
import math

class adam_optim:
    def __init__(self, model, lr, eps, device):
        self.m = OrderedDict()
        self.v = OrderedDict()
        self.beta_1=0.9 * torch.ones(1).to(device)
        self.beta_2=0.999 * torch.ones(1).to(device)
        self.eps = eps * torch.ones(1).to(device)
        self.lr = lr 
        for k, params in enumerate(model.parameters()):
            # if k<5:
            self.m['layer{}'.format(k)] = torch.zeros_like(params).to(device)
            self.v['layer{}'.format(k)] = torch.zeros_like(params).to(device)
    
    def update_params(self,model,t):
        m_key = list(self.m.keys())
        v_key = list(self.v.keys())
        for k, params in enumerate(model.parameters()):   
            if params.grad !=None: 
                self.m[m_key[k]] = self.beta_1 * self.m[m_key[k]] + (1-self.beta_1) * params.grad.data
                self.v[v_key[k]] = self.beta_2 * self.v[v_key[k]] + (1-self.beta_2) * torch.pow(params.grad.data,2)
                m_hat = self.m[m_key[k]] / (1-torch.pow(self.beta_1,t))
                v_hat = self.v[v_key[k]] / (1-torch.pow(self.beta_2,t))
                grad_mod = m_hat / (torch.sqrt(v_hat) + self.eps)
                params.data = params.data - self.lr * grad_mod

    def update_params_projected(self,model,t,feature_mat):
        m_key = list(self.m.keys())
        v_key = list(self.v.keys())
        kk = 0 
        for k, params in enumerate(model.parameters()):   
            if params.grad !=None:
                self.m[m_key[k]] = self.beta_1 * self.m[m_key[k]] + (1-self.beta_1) * params.grad.data
                self.v[v_key[k]] = self.beta_2 * self.v[v_key[k]] + (1-self.beta_2) * torch.pow(params.grad.data,2)
                m_hat = self.m[m_key[k]] / (1-torch.pow(self.beta_1,t))
                v_hat = self.v[v_key[k]] / (1-torch.pow(self.beta_2,t))
                grad_mod = m_hat / (torch.sqrt(v_hat) + self.eps)
                # gradient projection     
                sz =  params.grad.data.size(0)
                if k<4 : # First 4 layers (no Bias, no BN)
                    grad_mod = grad_mod - torch.mm(grad_mod.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())
                    params.data = params.data - self.lr * grad_mod
                    kk +=1
                else: # rest of the network (critique and action_dist)
                    params.data = params.data - self.lr * grad_mod
    
    def update_lr (self, lr):
        self.lr = lr

class adam_optim_bias:
    def __init__(self, model, lr, eps, device):
        self.m = OrderedDict()
        self.v = OrderedDict()
        self.beta_1=0.9 * torch.ones(1).to(device)
        self.beta_2=0.999 * torch.ones(1).to(device)
        self.eps = eps * torch.ones(1).to(device)
        self.lr = lr 
        for k, params in enumerate(model.parameters()):
            # if k<5:
            self.m['layer{}'.format(k)] = torch.zeros_like(params).to(device)
            self.v['layer{}'.format(k)] = torch.zeros_like(params).to(device)
    
    def update_params(self,model,t):
        m_key = list(self.m.keys())
        v_key = list(self.v.keys())
        for k, params in enumerate(model.parameters()):   
            if params.grad !=None: 
                self.m[m_key[k]] = self.beta_1 * self.m[m_key[k]] + (1-self.beta_1) * params.grad.data
                self.v[v_key[k]] = self.beta_2 * self.v[v_key[k]] + (1-self.beta_2) * torch.pow(params.grad.data,2)
                m_hat = self.m[m_key[k]] / (1-torch.pow(self.beta_1,t))
                v_hat = self.v[v_key[k]] / (1-torch.pow(self.beta_2,t))
                grad_mod = m_hat / (torch.sqrt(v_hat) + self.eps)
                params.data = params.data - self.lr * grad_mod

    def update_params_projected(self,model,t,feature_mat):
        m_key = list(self.m.keys())
        v_key = list(self.v.keys())
        kk = 0 
        for k, params in enumerate(model.parameters()):   
            if params.grad !=None:
                self.m[m_key[k]] = self.beta_1 * self.m[m_key[k]] + (1-self.beta_1) * params.grad.data
                self.v[v_key[k]] = self.beta_2 * self.v[v_key[k]] + (1-self.beta_2) * torch.pow(params.grad.data,2)
                m_hat = self.m[m_key[k]] / (1-torch.pow(self.beta_1,t))
                v_hat = self.v[v_key[k]] / (1-torch.pow(self.beta_2,t))
                grad_mod = m_hat / (torch.sqrt(v_hat) + self.eps)
                # gradient projection     
                sz =  params.grad.data.size(0)
                if k<8 and len(params.size())!=1: # First 4 layers (no BN)
                    grad_mod = grad_mod - torch.mm(grad_mod.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())
                    params.data = params.data - self.lr * grad_mod
                    kk +=1
                elif (k<8 and len(params.size())==1): # clear bias grads
                    params.grad.data.fill_(0)
                else: # rest of the network (critique and action_dist)
                    params.data = params.data - self.lr * grad_mod
    
    def update_lr (self, lr):
        self.lr = lr

