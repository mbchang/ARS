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
from policies import ARS_LinearAgent, ARS_MasterLinearAgent
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
                 organism_builder=None,
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
        self.worker_organism = organism_builder()
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
            action = self.worker_organism.forward(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break

        rollout_stats = AttrDict(
            total_reward=total_reward,
            steps=steps)

        return rollout_stats
            
    def evaluate_rollout(self, master_organism):
        # set to false so that evaluation rollouts are not used for updating state statistics
        self.worker_organism.evaluate_mode()
        self.worker_organism.sync_weights(master_organism)
        # for evaluation we do not shift the rewards (shift = 0) and we use the
        # default rollout length (1000 for the MuJoCo locomotion tasks)
        rollout_stats = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
        return rollout_stats

    def train_rollout(self, master_organism, shift):
        idx, delta = self.deltas.get_delta(self.worker_organism.get_num_weights())
        delta = self.delta_std * delta  # *= doesn't work!

        # set to true so that state statistics are updated 
        self.worker_organism.train_mode()

        # compute reward and number of timesteps used for positive perturbation rollout
        self.worker_organism.sync_weights(master_organism)
        self.worker_organism.add_noise_to_weights(delta)
        pos_rollout_stats = self.rollout(shift = shift)

        # compute reward and number of timesteps used for negative pertubation rollout
        self.worker_organism.sync_weights(master_organism)
        self.worker_organism.add_noise_to_weights(-delta)
        neg_rollout_stats = self.rollout(shift = shift)

        combined_rollout_stats = AttrDict(
            total_reward=[pos_rollout_stats['total_reward'], neg_rollout_stats['total_reward']],
            idx=idx,
            steps=pos_rollout_stats['steps']+neg_rollout_stats['steps']
            )
        return combined_rollout_stats



    def do_rollouts(self, master_organism, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        all_rollout_stats = []

        for i in range(num_rollouts):

            if evaluate:
                rollout_stats = self.evaluate_rollout(master_organism)
            else:
                rollout_stats = self.train_rollout(master_organism, shift)
            all_rollout_stats.append(rollout_stats)
        return all_rollout_stats
    

class ARS_Sampler(object):
    def __init__(self, num_deltas, shift,
        seed, 
        env_name,
        organism_builder,  # can look at this
        deltas_id,
        rollout_length,
        delta_std,
        num_workers,
        worker_builder=Worker):

        self.num_deltas = num_deltas
        self.shift = shift

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [worker_builder(seed + 7 * i,
                                      env_name=env_name,
                                      organism_builder=organism_builder,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        self.timesteps = 0

    def gather_experience(self, num_rollouts, evaluate, master_organism):
        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        results_one = [worker.do_rollouts(master_organism,
                                             num_rollouts = num_rollouts,
                                             shift = self.shift,
                                             evaluate=evaluate) for worker in self.workers]

        results_two = [worker.do_rollouts(master_organism,
                                             num_rollouts = 1,
                                             shift = self.shift,
                                             evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]
        return results_one + results_two

    def consolidate_experience(self, results, evaluate):
        rollout_rewards, deltas_idx = [], [] 

        for result in results:
            for subresult in result:
                if not evaluate:
                    self.timesteps += subresult['steps']
                    deltas_idx.append(subresult['idx'])
                else:
                    deltas_idx.append(-1)
                rollout_rewards.append(subresult['total_reward'])

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)  # (100,) for eval; (num_deltas, 2) for train
        return deltas_idx, rollout_rewards

    def update_master_from_workers(self, master_organism, workers):
        for worker in workers:
            master_organism.update_filter(worker.worker_organism)
        master_organism.stats_increment()

    def sync_workers_to_master(self, master_organism, workers):
        master_organism.clear_filter_buffer()
        # sync all workers
        for worker in workers:
            worker.worker_organism.sync_filter(master_organism)
        for worker in workers:
            worker.worker_organism.stats_increment()

    # note that this is all about syncing and updating the filter
    def sync_statistics(self, master_organism):
        t1 = time.time()
        # 1. sync master agent to workers
        self.update_master_from_workers(master_organism, self.workers)
        # 2. broadcast master agent to workers
        self.sync_workers_to_master(master_organism, self.workers)
        t2 = time.time()
        print('\tTime to sync statistics:', t2 - t1)


class ARSExperiment(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, 
                 # agent_args=None,
                 organism_builder=None,
                 logdir=None, 
                 params=None,
                 master_organism=None,
                 sampler_builder=None,
                 ):

        logz.configure_output_dir(logdir)
        logz.save_params(params)
        
        env = gym.make(params['env_name'])
        
        # self.action_size = env.action_space.shape[0]
        # self.ob_size = env.observation_space.shape[0]
        # self.num_deltas = params['n_directions']
        # self.deltas_used = params['deltas_used']
        self.logdir = logdir
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise_serial()
        self.deltas = SharedNoiseTable(deltas_id, seed = params['seed'] + 3)
        print('Created deltas table.')

        ########################################################

        self.master_organism = master_organism


        self.sampler = sampler_builder(
            num_deltas=params['n_directions'],
            shift=params['shift'],
            num_workers=params['n_workers'],
            seed=params['seed'],
            env_name=params['env_name'],
            # agent_args=agent_args,
            organism_builder=organism_builder,#lambda: ARS_LinearAgent(agent_args)
            deltas_id=deltas_id,
            rollout_length=params['rollout_length'],
            delta_std=params['delta_std'], 
            )

             # maybe we'd need to merge Sampler and Agent
        # agent holds the parameters, but sampler takes the agent and does the parallel rollouts
        # so agent should not have the workers at all...
        # agent should just contain the parameter.
        # but the sampler would need to take the agent in.
        # so the sampler is the thing that takes a single agent, and creates a bunch of workers
        # modeled the agent.

        self.rl_alg = ARS_RL_Alg(
            deltas=self.deltas,  # noise table
            num_deltas=params['n_directions'],  # N
            deltas_used=params['deltas_used']  # b
            )

        ########################################################

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """
        t1 = time.time()

        results = self.sampler.gather_experience(num_rollouts, evaluate, self.master_organism)
        deltas_idx, rollout_rewards = self.sampler.consolidate_experience(results, evaluate)

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
        self.master_organism.update(self.rl_alg, deltas_idx, rollout_rewards)
        self.sampler.sync_statistics(self.master_organism)
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
        np.savez(self.logdir + "/lin_policy_plus", self.master_organism.get_state())
        
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
    is_disc_action = len(env.action_space.shape) == 0
    ac_dim = env.action_space.n if is_disc_action else env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    agent_args={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSExperiment(
                     # agent_args=agent_args,
                     logdir=logdir,
                     params=params,
                     organism_builder=lambda: ARS_LinearAgent(agent_args),
                     master_organism=ARS_MasterLinearAgent(
                        agent_args=agent_args, 
                        step_size=params['step_size']),
                     sampler_builder=ARS_Sampler,
                     )

        
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

