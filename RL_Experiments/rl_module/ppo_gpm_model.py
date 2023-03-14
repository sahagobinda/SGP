import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from .a2c_ppo_acktr.utils import init, init_weight

# quant layer specific
from .quant_layer import Conv2d_Q, Linear_Q
import pdb
from copy import deepcopy
from collections import OrderedDict
import math

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    #Lin: input map size , output: output map_size
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                hidden_size = 1024
            elif len(obs_shape) == 1:
                base = MLPBase
                hidden_size = 64
            else:
                raise NotImplementedError

        self.features = base(obs_shape[0], **base_kwargs)
        self.init()

        # value critic for each task
        self.critic = nn.ModuleList()
        for _ in range(len(taskcla)):
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
            self.critic.append(init_(nn.Linear(hidden_size, 1)))

        # action distribution for each task
        self.dist = nn.ModuleList()
        for taskid, num_outputs in taskcla:
            self.dist.append(Categorical(self.features.output_size, num_outputs))

    @property
    def is_recurrent(self):
        return self.features.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.features.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, taskid, deterministic=False):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, taskid):
        actor_features, _ = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, taskid):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    # def fw_pass(self,inputs,collect_act=False):
    #     actor_features = self.features(inputs,collect_act)

    #     return actor_features

    def init (self):
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # print (m)
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
 

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=1024):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.stride=[]
        
        # init_ = lambda m: init_weight(m, nn.init.orthogonal_,nn.init.calculate_gain('relu'))

        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4,bias=False)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2,bias=False)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 128, 3, stride=1,bias=False)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(128 * 7 * 7, hidden_size,bias=False)), nn.ReLU())

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))

        # self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.map.append(84) #1
        self.ksize.append(8) #1
        self.in_channel.append(num_inputs)#1 
        self.stride.append(4) #1
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)
        s=compute_conv_output_size(84,8,4) # (map_size_input,k_size,stride)
        self.map.append(s) #2

        self.ksize.append(4) #2
        self.in_channel.append(32) #2
        self.stride.append(2) #2
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        s=compute_conv_output_size(s,4,2) # (map_size_input,k_size)
        self.map.append(s) #3

        self.ksize.append(3) #3
        self.in_channel.append(64) #3
        self.stride.append(1) #3
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, bias=False)
        s=compute_conv_output_size(s,3,1) # (map_size_input,k_size)
        self.map.append(s*s*128) #4

        self.fc1   = nn.Linear(128 * 7 * 7, hidden_size, bias=False)
        self.relu  = nn.ReLU()

        self.train()

    # def forward(self, inputs):#, rnn_hxs, masks):
    #     x = self.main(inputs / 255.0)

    #     # if self.is_recurrent:
    #     #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

    #     # return self.critic_linear(x), x, rnn_hxs
    #     return x #, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks, collect_act=False):
        bsz = deepcopy(inputs.size(0))
        x = inputs / 255.0
        if collect_act==True: self.act['conv1']=x
        x = self.relu(self.conv1(x))
        if collect_act==True: self.act['conv2']=x
        x = self.relu(self.conv2(x))
        if collect_act==True: self.act['conv3']=x
        x = self.relu(self.conv3(x))
        x = x.view(bsz,-1)
        if collect_act==True: self.act['fc1']=x
        x = self.relu(self.fc1(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        
        return x, rnn_hxs 


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        # return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
        return hidden_actor, rnn_hxs

class QPolicy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, base_kwargs=None):
        super(QPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = QCNNBase
                hidden_size = 1024
            elif len(obs_shape) == 1:
                base = QMLPBase
                hidden_size = 64
            else:
                raise NotImplementedError

        self.features = base(obs_shape[0], **base_kwargs)

        # value critic for each task
        self.critic = nn.ModuleList()
        for _ in range(len(taskcla)):
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
            self.critic.append(init_(nn.Linear(hidden_size, 1)))

        # action distribution for each task
        self.dist = nn.ModuleList()
        for taskid, num_outputs in taskcla:
            self.dist.append(Categorical(self.features.output_size, num_outputs))

    @property
    def is_recurrent(self):
        return self.features.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.features.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, taskid, deterministic=False):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, taskid):
        actor_features, _ = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, taskid):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class QCNNBase(NNBase):
    def __init__(self, num_inputs, F_prior, recurrent=False, hidden_size=1024):
        super(QCNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), nn.init.calculate_gain('relu'))

        # previous setup, the same as arch reported in EWC
        self.main = nn.Sequential(
            Conv2d_Q(num_inputs, 32, 8, stride=4, F_prior=F_prior), nn.ReLU(),
            Conv2d_Q(32, 64, 4, stride=2, F_prior=F_prior), nn.ReLU(),
            Conv2d_Q(64, 128, 3, stride=1, F_prior=F_prior), nn.ReLU(), Flatten(),
            Linear_Q(128 * 7 * 7, hidden_size, F_prior=F_prior), nn.ReLU())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return x, rnn_hxs


class QMLPBase(NNBase):
    def __init__(self, num_inputs, F_prior, recurrent=False, hidden_size=64):
        super(QMLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            Linear_Q(num_inputs, hidden_size, F_prior=F_prior), nn.Tanh(),
            Linear_Q(hidden_size, hidden_size, F_prior=F_prior), nn.Tanh())

        self.critic = nn.Sequential(
            Linear_Q(num_inputs, hidden_size, F_prior=F_prior), nn.Tanh(),
            Linear_Q(hidden_size, hidden_size, F_prior=F_prior), nn.Tanh())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return hidden_actor, rnn_hxs

# if __name__ == '__main__':
#     # main()
#     taskcla = [(0,14)]#, (1,18), (2,18), (3,18), (4,18), (5,6)]
#     task_sequences = [(0, 'KungFuMasterNoFrameskip-v4')]#, (1, 'BoxingNoFrameskip-v4'),
#                     # (2, 'JamesbondNoFrameskip-v4'), (3, 'KrullNoFrameskip-v4'),
#                     # (4, 'RiverraidNoFrameskip-v4'), (5, 'SpaceInvadersNoFrameskip-v4')]

#     # hard coded for atari environment
#     obs_shape = (4,84,84)
#     actor_critic = Policy(obs_shape,
#                           taskcla)
#     # base_kwargs={'recurrent': args.recurrent_policy})
#     for k, (m, param) in enumerate(actor_critic.named_parameters()):
#         print (m,param.shape)