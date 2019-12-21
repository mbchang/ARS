'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter
import optimizers
import torch
import torch.nn as nn


class Base_ARS_Agent(object):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self):
        super(Base_ARS_Agent, self).__init__()
        self.should_update_filter = True

    #########################################################
    # weight modifications

    def copy_weights(self):
        return np.copy(self.weights)

    def get_weight_shape(self):
        return self.weights.shape

    def get_num_weights(self):
        return self.weights.size

    def sync_weights(self, other_agent):
        self.weights[:] = other_agent.copy_weights()[:]
        return

    def add_noise_to_weights(self, noise_block):
        noise_block = noise_block.reshape(self.weights.shape)
        self.weights += noise_block
        # print(self.weights)
        # assert False

    def add_to_weights(self, added_weights):
        self.weights += added_weights

    #########################################################
    # observation filter stuff

    def evaluate_mode(self):
        self.should_update_filter = False

    def train_mode(self):
        self.should_update_filter = True

    def get_state(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def stats_increment(self):
        self.observation_filter.stats_increment()
        return

    def get_filter(self):
        return self.observation_filter

    def sync_filter(self, other):
        self.observation_filter.sync(other.get_filter())
        return
    #########################################################


class ARS_MasterAgent(Base_ARS_Agent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, step_size=0.1):
        Base_ARS_Agent.__init__(self)

        # Hmm. Maybe this shouldn't be here.
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.weights, step_size)        
        print("Initialization of ARS complete.")

    #########################################################
    # observation filter stuff
    def update_filter(self, other):
        # print('updating filter for agent {}'.format(self.id))
        self.observation_filter.update(other.get_filter())

    def clear_filter_buffer(self):
        self.observation_filter.clear_buffer()
    #########################################################/.    


class ARS_LinearAgent(Base_ARS_Agent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_args, id_num=0):
        Base_ARS_Agent.__init__(self)
        assert len(agent_args['ob_dim']) == 1
        self.ob_dim = agent_args['ob_dim'][0]
        self.ac_dim = agent_args['ac_dim']
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        # self.observation_filter = get_filter(agent_args['ob_filter'], shape = (self.ob_dim,))
        self.observation_filter = get_filter(agent_args['ob_filter'], shape = (self.ob_dim,))

        ####################################
        # auction stuff
        self.id = id_num
        self.active = True
        ####################################


    def forward(self, ob):
        ob = ob.ravel() # hacky way to deal with images  (H, W, C) for images, (D,) for features
        ob = self.observation_filter(ob, update=self.should_update_filter)
        action = np.dot(self.weights, ob)
        if len(action) == 1:
            action = action[0]
        # print('action', action)
        return action


# ########*****

class ARS_LinearAgent_PT(Base_ARS_Agent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_args, id_num=0):
        Base_ARS_Agent.__init__(self)
        self.ob_dim = agent_args['ob_dim']
        self.ac_dim = agent_args['ac_dim']

        self.layer = nn.Linear(self.ob_dim, self.ac_dim, bias=False)
        self.layer.weight.data.fill_(0)  # initialize
        self.weights = self.layer.weight.data  # seems like this is a reference not a copy

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(agent_args['ob_filter'], shape = (self.ob_dim,))

        ####################################
        # auction stuff
        self.id = id_num
        self.active = True
        ####################################

    #########################################################
    # weight modifications

    def get_state(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights.cpu().numpy(), mu, std])
        return aux

    def copy_weights(self):
        return np.copy(self.weights.cpu().numpy())

    def get_weight_shape(self):
        return self.weights.cpu().numpy().shape

    def get_num_weights(self):
        return self.weights.cpu().numpy().size

    def sync_weights(self, other_agent):
        self.weights[:] = torch.Tensor(other_agent.copy_weights()[:])  # probably really inefficient
        return

    def add_noise_to_weights(self, noise_block):
        noise_block = noise_block.reshape(self.get_weight_shape())
        self.weights += torch.Tensor(noise_block)

    def add_to_weights(self, added_weights):
        self.weights += torch.Tensor(added_weights)

    #########################################################

    def forward(self, ob):
        ob = ob.ravel() # hacky way to deal with images  (H, W, C) for images, (D,) for features
        ob = self.observation_filter(ob, update=self.should_update_filter)

        # ************************************** #
        with torch.no_grad():
            action = self.layer(torch.Tensor(ob)).cpu().numpy()
         # ************************************** #
        if len(action) == 1:
            action = action[0]
        return action


class ARS_Conv2dAgent_PT(ARS_LinearAgent_PT):
    def __init__(self, agent_args, id_num=0):

        """
        need to change observation_filter
        need to change layer
        """
        self.ob_dim = agent_args['ob_dim']  # (H, W, C)
        self.ob_h, self.ob_w, self.ob_c = self.ob_dim
        self.ac_dim = agent_args['ac_dim']
        assert self.ob_h == self.ob_w

        self.layer = nn.Conv2d(
            in_channels=self.ob_c,
            out_channels=self.ac_dim,
            kernel_size=self.ob_h)

        self.weights = self.layer.weight.data  # (out_channels, in_channels, kernel_size, kernel_size)

        self.observation_filter = get_filter(
            agent_args['ob_filter'], 
            shape=(self.ob_c, self.ob_h, self.ob_w))

        ####################################
        # auction stuff
        self.id = id_num
        self.active = True
        ####################################

    def forward(self, ob):
        """
        need to reshape I think
        not raveling anymore
        """
        ob = torch.Tensor([ob])
        # (bsize, H, W, C) --> (bsize, C, H, W)
        ob = ob.transpose(1, 3).transpose(2, 3)
        with torch.no_grad():
            action = self.layer(ob)
            if self.ac_dim == 1:
                action = action.item()
            else:
                assert False
        return action

########*****

"""
I think the interface should be:
- the worker provides the noise
- the agent takes the noise and updates its own weights


0. noise = worker.get_noise(worker.policy.get_num_weights())
1. worker.policy.update_filter = True
2. worker.policy.sync_weights(master_policy)
3. worker.policy.add_noise_to_weights(noise)
4. worker.rollout()
5. worker.policy.sync_weights(master_policy)
6. worker.policy.add_noise_to_weights(-noise)
7. worker.rollout()
"""                                                                     
class ARS_MasterLinearAgent(ARS_LinearAgent, ARS_MasterAgent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_args, id_num=0, step_size=0.1):
        ARS_LinearAgent.__init__(self, agent_args, id_num)
        ARS_MasterAgent.__init__(self, step_size)

    def update(self, rl_alg, deltas_idx, rollout_rewards):
        rl_alg.improve(self, deltas_idx, rollout_rewards)



class ARS_MasterConv2dAgent_PT(ARS_Conv2dAgent_PT, ARS_MasterAgent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_args, id_num=0, step_size=0.1):
        ARS_Conv2dAgent_PT.__init__(self, agent_args, id_num)
        ARS_MasterAgent.__init__(self, step_size)

    def update(self, rl_alg, deltas_idx, rollout_rewards):
        rl_alg.improve(self, deltas_idx, rollout_rewards)


"""
Ok, it seems like the only difference between the MasterAgent and the WorkerAgent is that the 
MasterAgent has the optimizer.
"""








