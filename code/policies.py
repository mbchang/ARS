'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter
import optimizers


class Policy(object):

    def __init__(self, agent_params):

        self.ob_dim = agent_params['ob_dim']
        self.ac_dim = agent_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(agent_params['ob_filter'], shape = (self.ob_dim,))
        self.should_update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_observation_filter(self):
        return self.observation_filter

    def forward(self, ob):
        raise NotImplementedError

    def copy_weights(self):
        raise NotImplementedError

class LinearAgent(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_params):
        Policy.__init__(self, agent_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def forward(self, ob):
        ob = self.observation_filter(ob, update=self.should_update_filter)
        return np.dot(self.weights, ob)

    def get_state(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def sync_weights(self, master_agent):
        self.update_weights(master_agent.copy_weights())

    def add_noise_to_weights(self, noise_block):
        noise_block = noise_block.reshape(self.weights.shape)
        self.weights += noise_block

    def get_weight_shape(self):
        return self.weights.shape

    def get_num_weights(self):
        return self.weights.size

    def evaluate_mode(self):
        self.should_update_filter = False

    def train_mode(self):
        self.should_update_filter = True

    def stats_increment(self):
        self.observation_filter.stats_increment()
        return

    def get_filter(self):
        return self.observation_filter

    def sync_filter(self, other):
        self.observation_filter.sync(other.get_filter())
        return


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

        

class MasterLinearAgent(LinearAgent):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, agent_params, step_size=0.1):
        LinearAgent.__init__(self, agent_params)

        # Hmm. Maybe this shouldn't be here.
        # initialize optimization algorithm
        self.step_size = step_size
        self.optimizer = optimizers.SGD(self.weights, self.step_size)        
        print("Initialization of ARS complete.")

    def copy_weights(self):
        return np.copy(self.weights)

    def update_filter(self, other_agent):
        self.observation_filter.update(other_agent.get_filter())

    def clear_filter_buffer(self):
        self.observation_filter.clear_buffer()








