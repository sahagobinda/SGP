import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
from .adam_custom import adam_optim, adam_optim_bias

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    #Lin: input map size , output: output map_size
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class PPO_SGP():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 threshold = 0.97,
                 threshold_inc = 0.005,
                 custom_adam=False,
                 gpm_mini_batch=32,
                 scale_coff=5,
                 network_bias=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # self.ewc_epoch = 1
        # self.ewc_lambda = ewc_lambda
        
        # print ('ewc_lambda : ', self.ewc_lambda)

        # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.lr = lr
        self.eps = eps
        self.gpm_mini_batch = gpm_mini_batch
        self.custom_adam = custom_adam
        self.scale_coff = scale_coff
        self.network_bias = network_bias
        self.iteration = 0
        print ('custom_adam : ',self.custom_adam)
        print ('scale_coff : ',self.scale_coff)
        print ('network_bias : ',self.network_bias)
        # self.EWC_task_count = 0
        # self.divide_factor = 0
        # self.online = online

        # GPM related 
        self.threshold =  np.array([threshold] * 4) 
        self.threshold_inc = np.array([threshold_inc] * 4)
        self.feature_list = []
        self.importance_list = []
        self.feature_mat = []

    def renew_optimizer(self,device):
        if self.custom_adam == True:
            if self.network_bias == True:
                self.optimizer = adam_optim_bias(self.actor_critic, lr=self.lr, eps=self.eps, device=device)
            else:   
                self.optimizer = adam_optim(self.actor_critic, lr=self.lr, eps=self.eps, device=device)
            self.iteration = 0 
        else:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)

    
    def compute_feature_mat (self, device):
        self.feature_mat = []
        # Projection Matrix Precomputation
        print ('-'*40)
        for i in range(len(self.feature_list)):
            Uf=torch.Tensor(np.dot(self.feature_list[i],np.dot(np.diag(self.importance_list[i]),self.feature_list[i].transpose()))).to(device)
            print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            Uf.requires_grad = False
            self.feature_mat.append(Uf)
        print ('-'*40)
    
    def update(self, rollouts, task_num):
        # compute advantage function 
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if self.custom_adam == False: 
                    self.optimizer.zero_grad()

                # reg_loss = self.ewc_lambda * self.ewc_loss()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                
                # for k, (m, param) in enumerate(self.actor_critic.named_parameters()):
                #     if param.grad != None:
                #         print (m,param.shape,param.grad.shape)

                # pdb.set_trace()
                ## Do network update with ADAM/CUSTOM_ADAM
                if self.custom_adam == True:
                    # self.iteration +=1
                    if task_num == 0:
                        self.optimizer.update_params(self.actor_critic,self.iteration)
                    elif task_num >0:
                        self.optimizer.update_params_projected(self.actor_critic,self.iteration,self.feature_mat)
                    self.actor_critic.zero_grad() # model zero grad

                else:
                    ## Implement GPM - Gradient Projections 
                    if task_num>0:
                        kk = 0 
                        for k, (m,params) in enumerate(self.actor_critic.named_parameters()):
                            if k<8 and len(params.size())!=1: # 1st 4 layers 
                                sz =  params.grad.data.size(0)
                                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                                        self.feature_mat[kk]).view(params.size())
                                kk +=1
                            elif (k<8 and len(params.size())==1): # clear bias grads
                                params.grad.data.fill_(0)
                           
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def get_representation_matrix (self, rollouts, task_num, collect_act=True):

        mat_list=[[],[],[],[]] ## return this contains layerwise (4-layer) activation for GPM 
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        # set number of mini-batch to 32
        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, self.gpm_mini_batch) #32
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, self.gpm_mini_batch) #32

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, _, \
               _, _, masks_batch, _, _ = sample

            batch_size_t = obs_batch.shape[0]
            # total_batch += batch_size_t

            # clear gradient
            self.actor_critic.zero_grad()

            # get action distribution (collect samples in forward pass)
            actor_features, _ = self.actor_critic.features(obs_batch, 
                recurrent_hidden_states_batch, masks_batch,collect_act)

            ## Get representation matrix for GPM 
            act_key=list(self.actor_critic.features.act.keys())
            for i in range(len(self.actor_critic.features.map)):
                bsz = batch_size_t 
                k=0
                if i<3:
                    ksz= self.actor_critic.features.ksize[i]
                    s=compute_conv_output_size(self.actor_critic.features.map[i],self.actor_critic.features.ksize[i],self.actor_critic.features.stride[i])
                    mat = np.zeros((self.actor_critic.features.ksize[i]*self.actor_critic.features.ksize[i]*self.actor_critic.features.in_channel[i],s*s*bsz))
                    act = self.actor_critic.features.act[act_key[i]].detach().cpu().numpy()
                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                                k +=1
                    # mat_list.append(mat)
                else:
                    act = self.actor_critic.features.act[act_key[i]].detach().cpu().numpy()
                    mat = act[0:bsz].transpose()
                    # mat_list.append(mat)

                # store activations 
                try:
                    mat_list[i]  = np.concatenate((mat_list[i],  mat), axis=1)
                except:
                    mat_list[i]  = mat

        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_list)):
            print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
        print('-'*30)

        return mat_list


    def get_SGP(self, mat_list, task_id, log_dir):
        threshold = self.threshold + task_id * self.threshold_inc
        feature_list = self.feature_list
        importance_list = self.importance_list
        print ('Threshold: ', threshold) 
        # plt.figure(figsize=(10, 6))
        if not feature_list:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-1)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold[i])+1  
                feature_list.append(U[:,0:r])
                importance = ((self.scale_coff+1)*S[0:r])/(self.scale_coff*S[0:r]+ max(S[0:r])) 
                importance_list.append(importance)
                # plt.plot(importance_list[i], label='Layer {}'.format(i+1))
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-4)
                act_proj = np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                r_old = feature_list[i].shape[1]
                Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
                importance_new_on_old = np.dot(np.dot(feature_list[i].transpose(),Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
                importance_new_on_old = np.sqrt(importance_new_on_old) # surrogate singular values along old GPM basis 
                
                act_hat = activation - act_proj
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-5)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
                
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    # update importances 
                    importance = importance_new_on_old
                    importance = ((self.scale_coff+1)*importance)/(self.scale_coff*importance+ max(importance)) 
                    importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
                    importance_list[i] = importance # update importance
                    # plt.plot(importance_list[i],label='Layer {}'.format(i+1))
                    continue

                # Compute importances
                importance = np.hstack((importance_new_on_old,S[0:r]))
                importance = ((self.scale_coff+1) * importance)/(self.scale_coff * importance + max(importance))         
                # importance update 
                importance [0:r_old] = np.clip(importance [0:r_old] + importance_list[i][0:r_old], 0, 1) 

                # update GPM
                Ui=np.hstack((feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    feature_list[i]=Ui[:,0:Ui.shape[0]]
                    importance_list[i] = importance[0:Ui.shape[0]]
                else:
                    feature_list[i]=Ui
                    importance_list[i] = importance 
                # plt.plot(importance_list[i],label='Layer {}'.format(i+1))

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(feature_list)):
            print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
        print('-'*40)

        return feature_list, importance_list  


    def update_SGP(self, rollouts, task_num, log_dir):
        mat_list = self.get_representation_matrix(rollouts, task_num)
        self.feature_list, self.importance_list = self.get_SGP(mat_list, task_num, log_dir)
