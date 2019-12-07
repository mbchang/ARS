'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import parser
import time
import os
import numpy as np
import gym
import logz
# import ray
import utils
import optimizers
from policies import *
# import socket
from shared_noise import *

class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    


class ARS_Sampler(object):
    def __init__(self, num_deltas, num_workers, shift, workers):
        self.num_deltas = num_deltas
        self.num_workers = num_workers
        self.shift = shift

        self.workers = workers

    def gather_experience(self, num_rollouts, evaluate,
        w_policy):
        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        results_one = [worker.do_rollouts(np.copy(w_policy),
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        results_two = [worker.do_rollouts(np.copy(w_policy),
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]
        return results_one + results_two

    def consolidate_experience(self, results, evaluate,
        timesteps):
        rollout_rewards, deltas_idx = [], [] 

        for result in results:
            if not evaluate:
                timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        return deltas_idx, rollout_rewards, timesteps


class ARS_RL_Alg(object):
    def __init__(self, optimizer, deltas, num_deltas, deltas_used):
        self.optimizer = optimizer
        self.deltas = deltas
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used


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

    def update(self, deltas_idx, rollout_rewards, w_policy):
        t1 = time.time()
        g_hat = self.compute_error(deltas_idx, rollout_rewards, w_policy)
        print("\tEuclidean norm of update step:", np.linalg.norm(g_hat))

        w_policy -= self.optimizer._compute_step(g_hat).reshape(w_policy.shape)

        t2 = time.time()
        print('\t\tTime to update weights', t2-t1)
        return w_policy


class ARSExperiment(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320,  # N
                 deltas_used=320,  # b
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)
        
        env = gym.make(env_name)
        
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise_serial()
        self.deltas = SharedNoiseTable(deltas_id, seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

        ########################################################
        self.sampler = ARS_Sampler(
            num_deltas=self.num_deltas, 
            num_workers=self.num_workers,
            shift=self.shift,
            workers=self.workers)

        self.rl_alg = ARS_RL_Alg(
            optimizer=self.optimizer,
            deltas=self.deltas,
            num_deltas=self.num_deltas,
            deltas_used=self.deltas_used)
        ########################################################

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """
        t1 = time.time()

        results = self.sampler.gather_experience(
            num_rollouts, evaluate, self.w_policy)
        deltas_idx, rollout_rewards, self.timesteps = self.sampler.consolidate_experience(
            results, evaluate, self.timesteps)

        print('\tMaximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('\t\tTime to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards
        else:
            return deltas_idx, rollout_rewards

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        deltas_idx, rollout_rewards = self.aggregate_rollouts()
        self.w_policy = self.rl_alg.update(deltas_idx, rollout_rewards, self.w_policy)
        return

    def main_loop(self, num_iter):

        start = time.time()
        for i in range(num_iter):
            print('iter ', i,' start')
            
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('\tTotal time of one step', t2 - t1)           
            
            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):
                self.record_statistics(start, i)
                if self.params['debug']:
                    return  # for debugging

            self.sync_statistics()
            print('iter ', i,' done')
                        
        return 

    def record_statistics(self, start, i):
        rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
        w = self.workers[0].get_weights_plus_stats()
        np.savez(self.logdir + "/lin_policy_plus", w)
        
        print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        logz.log_tabular("AverageReward", np.mean(rewards))
        logz.log_tabular("StdRewards", np.std(rewards))
        logz.log_tabular("MaxRewardRollout", np.max(rewards))
        logz.log_tabular("MinRewardRollout", np.min(rewards))
        logz.log_tabular("timesteps", self.timesteps)
        logz.dump_tabular()

    def sync_statistics(self):
        t1 = time.time()
        # get statistics from all workers
        for j in range(self.num_workers):
            self.policy.observation_filter.update(self.workers[j].get_filter())
        self.policy.observation_filter.stats_increment()

        # make sure master filter buffer is clear
        self.policy.observation_filter.clear_buffer()
        # sync all workers
        for worker in self.workers:
            worker.sync_filter(self.policy.observation_filter)
        for worker in self.workers:
            worker.stats_increment()

        t2 = time.time()
        print('\tTime to sync statistics:', t2 - t1)

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSExperiment(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    ARS.main_loop(params['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(redis_address= local_ip + ':6379')
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

