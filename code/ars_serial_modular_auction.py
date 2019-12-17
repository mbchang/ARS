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
####################### AUCTION ########################
from policies_auction import ARS_LinearAuction, ARS_MasterLinearAuction
####################### AUCTION ########################
from shared_noise import *

from rl_alg import ARS as ARS_RL_Alg
from collections import OrderedDict


"""
Assumptions:
- gamma = 1
- fixed horizon length
- continuous control

- ok, here's a problem: can ARS work if the actions are bounded between 0 and 1?
- I suppose I can do sigmoid?
- well actaully since there is only a single nonlinear transformation at the very end
- you can just think of the sigmoid as part of the environment.

- ok the only thing left to consider is to how to constrain the output of the agents
"""

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__    

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
                 agent_args = None,
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

        ####################### AUCTION ########################
        self.worker_organism  = create_auction(
            agent_args, ARS_LinearAuction, ARS_LinearAgent)
        ####################### AUCTION ######################
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

    ####################### AUCTION ########################

    # good
    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0
        # ********************* AUCTION ********************* #
        agent_episode_data = {a.id: [] for a in self.worker_organism.get_active_agents()}
        episode_data = []
        # ********************* AUCTION ********************* #

        ob = self.env.reset()
        for i in range(rollout_length):
            action, winner, bids = self.worker_organism.forward(ob)
            next_ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            # ********************* AUCTION ********************* #
            for agent_id in bids.keys():
                agent_episode_data[agent_id].append(
                    AttrDict(state=ob, action=bids[agent_id])  # note that we assume fixed length so we do not care about the mask!
                    )
            episode_data.append(
                AttrDict(state=ob, winner=winner, next_state=next_ob, reward=reward))
            # ********************* AUCTION ********************* #
            if done:
                break
            ob = next_ob

        # ********************* AUCTION ********************* #
        # now compute payoffs of the agent
        agent_episode_data = self.record_payoffs(
            self.worker_organism, agent_episode_data, episode_data)

        # now compute the total reward for each agent
        agent_rollout_stats = AttrDict()
        for a_id in agent_episode_data:
            agent_rollout_stats[a_id] = AttrDict(
                total_reward=sum(s.payoff for s in agent_episode_data[a_id]),
                steps=len(agent_episode_data[a_id])
                )
        # ********************* AUCTION ********************* #
        return agent_rollout_stats

    def record_payoffs(self, worker_organism, agent_episode_data, society_episode_data):
        for t in range(len(society_episode_data)):
            state = society_episode_data[t].state
            winner = society_episode_data[t].winner
            bids = OrderedDict([(a.id, agent_episode_data[a.id][t]['action']) for a in worker_organism.get_active_agents()])  

            if t < len(society_episode_data)-1:
                next_winner = society_episode_data[t+1].winner
                next_winner_bid = agent_episode_data[next_winner][t+1].action
                next_state = society_episode_data[t+1].state
            else:
                next_winner_bid = 0
                next_state = society_episode_data[t].next_state

            reward = society_episode_data[t].reward  # not t+1!

            payoffs = worker_organism._compute_payoffs(state, bids, winner, next_winner_bid, next_state, reward)

            for agent in worker_organism.get_active_agents():
                agent_episode_data[agent.id][t].payoff = payoffs[agent.id]  # this line can be streamlined

        return agent_episode_data

    ####################### AUCTION ########################

    ####################### AUCTION ########################
    def evaluate_rollout(self, master_organism):
        # set to false so that evaluation rollouts are not used for updating state statistics
        self.worker_organism.evaluate_mode()
        self.worker_organism.sync_weights(master_organism)  # good

        # for evaluation we do not shift the rewards (shift = 0) and we use the
        # default rollout length (1000 for the MuJoCo locomotion tasks)
        rollout_stats = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
        
        return rollout_stats


    def train_rollout(self, master_organism, shift):
        agent_idx = {}
        agent_delta = {}

        for agent in self.worker_organism.agents:
            agent_idx[agent.id], agent_delta[agent.id] = self.deltas.get_delta(agent.get_num_weights())
            agent_delta[agent.id] = self.delta_std * agent_delta[agent.id]  # *= doesn't work!


        # set to true so that state statistics are updated 
        self.worker_organism.train_mode()

        # compute reward and number of timesteps used for positive perturbation rollout
        self.worker_organism.sync_weights(master_organism)
        for agent in self.worker_organism.agents:
            agent.add_noise_to_weights(agent_delta[agent.id])
        pos_rollout_stats = self.rollout(shift=shift)

        # compute reward and number of timesteps used for negative pertubation rollout
        self.worker_organism.sync_weights(master_organism)
        for agent in self.worker_organism.agents:
            agent.add_noise_to_weights(-agent_delta[agent.id])
        neg_rollout_stats = self.rollout(shift=shift)

        # combine the rollout_stats
        combined_rollout_stats = {}
        for agent_id in pos_rollout_stats.keys():
            combined_rollout_stats[agent_id] = AttrDict(
                total_reward=[pos_rollout_stats[agent_id]['total_reward'], 
                        neg_rollout_stats[agent_id]['total_reward']],
                idx=agent_idx[agent_id],
                steps=pos_rollout_stats[agent_id]['steps']+neg_rollout_stats[agent_id]['steps'])

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

    ####################### AUCTION ########################

    

class ARS_Sampler(object):
    def __init__(self, num_deltas, shift,
        seed, 
        env_name,
        agent_args,  # can look at this
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
                                      agent_args=agent_args,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        self.timesteps = 0

    ####################### AUCTION ########################
    # no change
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
    ####################### AUCTION ########################

    ####################### AUCTION ########################
    def consolidate_experience(self, results, evaluate):
        """
            results: 
                list of length num_workers
                    dict with keys: {'deltas_idx', 'rollout_rewards', 'steps'}
                        dict with keys: agent_id
                            list of length num_rollouts_per_worker
            output:
                deltas_idx or rollout_rewards or steps
                    dict with keys: agent_id
                        list of length num_rollouts

            you can consider how you would structure the information beforehand
            in order 
        """
        agent_ids = [a.id for a in self.workers[0].worker_organism.agents]
        agent_rollout_rewards = {a_id: [] for a_id in agent_ids}
        agent_deltas_idx = {a_id: [] for a_id in agent_ids}

        for result in results:
            for subresult in result:
                if not evaluate:
                    self.timesteps += subresult[0]['steps']
                for agent_id in agent_ids:
                    if not evaluate:
                        agent_deltas_idx[agent_id].append(subresult[agent_id]['idx'])
                    else:
                        agent_deltas_idx[agent_id].append(-1)
                    agent_rollout_rewards[agent_id].append(subresult[agent_id]['total_reward'])

        for agent_id in agent_ids:
            agent_deltas_idx[agent_id] = np.array(agent_deltas_idx[agent_id])
            agent_rollout_rewards[agent_id] = np.array(agent_rollout_rewards[agent_id], dtype=np.float64)

        return agent_deltas_idx, agent_rollout_rewards

    # actually to be honest you can consolidate_experience_auction for each agent in the outer loop

    ####################### AUCTION ########################


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


def create_auction(agent_args, auction_builder, agent_builder):
    import copy
    action_dim = agent_args['ac_dim']
    print('action_dim', action_dim)
    redundancy = 2
    subagent_args = copy.deepcopy(agent_args)
    subagent_args['ac_dim'] = 1
    agents = [agent_builder(subagent_args, id_num=i) for i in range(action_dim*redundancy)]
    return auction_builder(agents, action_dim)


class ARSExperiment(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 agent_args=None,
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


        ####################### AUCTION ########################
        self.master_organism  = create_auction(
            agent_args, ARS_MasterLinearAuction, ARS_MasterLinearAgent)
        # test this now.
        # this could definitely be refactored
        ####################### AUCTION ########################

        self.master_agent = ARS_MasterLinearAgent(
            agent_args=agent_args, 
            step_size=step_size)

        self.sampler = ARS_Sampler(
            num_deltas=self.num_deltas, 
            shift=self.shift,
            num_workers=num_workers,
            seed=seed, 
            env_name=env_name, 
            agent_args=agent_args, 
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

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """
        t1 = time.time()

        # results = self.sampler.gather_experience(num_rollouts, evaluate, self.master_agent)
        # deltas_idx, rollout_rewards = self.sampler.consolidate_experience(results, evaluate)

        ####################### AUCTION ########################
        """
        what does it mean for the sampler to gather_experience
        and consolidate  experience witht the auction?

        """
        results_auction = self.sampler.gather_experience(
            num_rollouts, evaluate, self.master_organism)
        deltas_idx_auction, rollout_rewards_auction = self.sampler.consolidate_experience(results_auction, evaluate)


        # assert False

        ####################### AUCTION ########################
        for agent_id in rollout_rewards_auction:
            print('\tMaximum reward of collected rollouts for agent {}: {}'.format( agent_id, rollout_rewards_auction[agent_id].max()))
        t2 = time.time()

        print('\t\tTime to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards_auction
        else:
            return deltas_idx_auction, rollout_rewards_auction

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        # import ipdb
        # ipdb.set_trace()
        deltas_idx, rollout_rewards = self.aggregate_rollouts()

        # actually this interface seems to make sense.
        self.master_organism.update(self.rl_alg, deltas_idx, rollout_rewards)
        self.sampler.sync_statistics(self.master_organism)

        # 12/15: 11:56pm stack pointer here. Seems that things can compiile. 
        # now it's just a matter if things are correct.
        # assert False
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
        np.savez(self.logdir + "/lin_policy_plus", self.master_agent.get_state())
        
        print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        for agent_id in rewards:
            logz.log_tabular("Agent {} AverageReward".format(agent_id), np.mean(rewards[agent_id]))
            logz.log_tabular("Agent {} StdRewards".format(agent_id), np.std(rewards[agent_id]))
            logz.log_tabular("Agent {} MaxRewardRollout".format(agent_id), np.max(rewards[agent_id]))
            logz.log_tabular("Agent {} MinRewardRollout".format(agent_id), np.min(rewards[agent_id]))

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
    agent_args={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSExperiment(env_name=params['env_name'],
                     agent_args=agent_args,
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

