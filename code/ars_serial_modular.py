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
import utils
import optimizers
from policies import *
from shared_noise import *

from rl_alg import ARS as ARS_RL_Alg

class Worker(object):
    """ 
    Object class for parallel rollout generation.

    Each worker gets
        - env seed
        - copy of the environment
        - noise
        - policy_config
        - rollout_length
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

        ################################################
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearAgent(policy_params)
        else:
            raise NotImplementedError
        # ---
        # this should be replaced by an agent builder
        # or you could literally just pass in a clone.
        ################################################
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

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

    def evaluate_rollout(self, w_policy):
        self.policy.update_weights(w_policy)
        
        # set to false so that evaluation rollouts are not used for updating state statistics
        self.policy.update_filter = False

        # for evaluation we do not shift the rewards (shift = 0) and we use the
        # default rollout length (1000 for the MuJoCo locomotion tasks)
        reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
        return reward, -1

    def train_rollout(self, w_policy, shift):
        idx, delta = self.deltas.get_delta(w_policy.size)
     
        delta = (self.delta_std * delta).reshape(w_policy.shape)

        # set to true so that state statistics are updated 
        self.policy.update_filter = True

        # compute reward and number of timesteps used for positive perturbation rollout
        self.policy.update_weights(w_policy + delta) 
        pos_reward, pos_steps  = self.rollout(shift = shift)

        # compute reward and number of timesteps used for negative pertubation rollout
        self.policy.update_weights(w_policy - delta)
        neg_reward, neg_steps = self.rollout(shift = shift) 

        return [pos_reward, neg_reward], idx, pos_steps + neg_steps


    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                rollout_reward, rollout_idx = self.evaluate_rollout(w_policy)
            else:
                rollout_reward, rollout_idx, rollout_steps = self.train_rollout(w_policy, shift)
                steps += rollout_steps

            rollout_rewards.append(rollout_reward)
            deltas_idx.append(rollout_idx)

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
    def __init__(self, num_deltas, shift,
        seed, 
        env_name,
        policy_params,  # can look at this
        deltas_id,
        rollout_length,
        delta_std,
        num_workers):

        self.num_deltas = num_deltas
        self.shift = shift

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        self.timesteps = 0

    def gather_experience(self, num_rollouts, evaluate, agent):
        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        results_one = [worker.do_rollouts(agent.copy(),
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        results_two = [worker.do_rollouts(agent.copy(),
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]
        return results_one + results_two

    def consolidate_experience(self, results, evaluate):
        rollout_rewards, deltas_idx = [], [] 

        for result in results:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        return deltas_idx, rollout_rewards


    def sync_statistics(self, agent):
        t1 = time.time()
        # get statistics from all workers
        for j in range(self.num_workers):
            agent.observation_filter.update(self.workers[j].get_filter())
        agent.observation_filter.stats_increment()

        # make sure master filter buffer is clear
        agent.observation_filter.clear_buffer()
        # sync all workers
        for worker in self.workers:
            worker.sync_filter(agent.observation_filter)
        for worker in self.workers:
            worker.stats_increment()

        t2 = time.time()
        print('\tTime to sync statistics:', t2 - t1)


"""
Your worker will be a copy of the organism, whether the organism is a society or agent.
This means that the worker should not know about the policy.
You need methods to clone the agent and update the agent.
    - let's implement that now.
    - the agent needs a method to copy itself
    - the agent needs a method to update weights
    - the agent needs a method to get weights

Ok, let's combine ARS_Agent with Policy.
Let's just make everything interface with Agent now.

"""



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

        ########################################################

        self.agent = MasterLinearAgent(
            policy_params=policy_params, 
            step_size=step_size)

        self.sampler = ARS_Sampler(
            num_deltas=self.num_deltas, 
            shift=self.shift,
            num_workers=num_workers,
            seed=seed, 
            env_name=env_name, 
            policy_params=policy_params, 
            deltas_id=deltas_id, 
            rollout_length=rollout_length, 
            delta_std=delta_std, 

            )  # maybe we'd need to merge Sampler and Agent
        # agent holds the parameters, but sampler takes the agent and does the parallel rollouts
        # so agent should not have the workers at all...
        # agent should just contain the parameter.
        # but the sampler would need to take the agent in.
        # so the sampler is the thing that takes a single agent, and creates a bunch of workers
        # modeled the agent.

        self.rl_alg = ARS_RL_Alg(
            deltas=self.deltas,  # noise table
            num_deltas=self.num_deltas,  # N
            deltas_used=self.deltas_used  # b
            )
        ########################################################

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """
        t1 = time.time()

        results = self.sampler.gather_experience(num_rollouts, evaluate, self.agent)
        deltas_idx, rollout_rewards = self.sampler.consolidate_experience(results, evaluate)

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
        # actually this interface seems to make sense.
        self.rl_alg.improve(self.agent, deltas_idx, rollout_rewards)
        self.sampler.sync_statistics(self.agent)
        return

    def main_loop(self, num_iter):
        start = time.time()
        for i in range(num_iter):
            print('iter ', i)  

            # record statistics every 10 iterations
            if (i % 2 == 0):
                self.eval_step(start, i)
                if self.params['debug'] and i == 4:
                    return  # for debugging

            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('\tTotal time of one step', t2 - t1)           
        return 

    def eval_step(self, start, i):
        rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
        np.savez(self.logdir + "/lin_policy_plus", self.agent.get_state())
        
        print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        logz.log_tabular("AverageReward", np.mean(rewards))
        logz.log_tabular("StdRewards", np.std(rewards))
        logz.log_tabular("MaxRewardRollout", np.max(rewards))
        logz.log_tabular("MinRewardRollout", np.min(rewards))
        logz.log_tabular("timesteps", self.sampler.timesteps)
        logz.dump_tabular()

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

    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

