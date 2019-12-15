'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter
import optimizers


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
        self.ob_dim = agent_args['ob_dim']
        self.ac_dim = agent_args['ac_dim']
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(agent_args['ob_filter'], shape = (self.ob_dim,))

        ####################################
        # auction stuff
        self.id = id_num
        self.active = True
        ####################################


    def forward(self, ob):
        ob = self.observation_filter(ob, update=self.should_update_filter)
        action = np.dot(self.weights, ob)
        if len(action) == 1:
            action = action[0]
        return action

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


"""
Ok, it seems like the only difference between the MasterAgent and the WorkerAgent is that the 
MasterAgent has the optimizer.
"""








