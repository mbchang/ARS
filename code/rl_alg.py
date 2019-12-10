import numpy as np
import time

import utils

class RandomSearchRlAlg():
    def __init__(self, deltas, num_deltas, deltas_used):
        self.deltas = deltas
        self.num_deltas = num_deltas  # N
        self.deltas_used = deltas_used  # b

"""
Ok I think this interface makes sense.
The only thing to change would be to use the replay buffer in the agent.
"""
class ARS(RandomSearchRlAlg):
    def __init__(self, deltas, num_deltas, deltas_used):
        super(ARS, self).__init__(deltas, num_deltas, deltas_used)

    def compute_error(self, deltas_idx, rollout_rewards, w_policy):
        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        return g_hat

    def improve(self, agent, deltas_idx, rollout_rewards):
        t1 = time.time()
        g_hat = self.compute_error(deltas_idx, rollout_rewards, agent.weights)
        print("\tEuclidean norm of update step:", np.linalg.norm(g_hat))
        agent.weights -= agent.optimizer._compute_step(g_hat).reshape(agent.weights.shape)
        t2 = time.time()
        print('\t\tTime to update weights', t2-t1)