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
from ars_serial_modular import ARS_Sampler as Base_ARS_Sampler
from ars_serial_modular import ARSExperiment as Base_ARSExperiment
from ars_serial_modular import Worker as Base_Worker
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

class Worker(Base_Worker):
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
        super(Worker, self).__init__(
            env_seed=env_seed,
            env_name=env_name,
            organism_builder=organism_builder,
            deltas=deltas,
            rollout_length=rollout_length,
            delta_std=delta_std)

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
                AttrDict(state=ob, winner=winner, next_state=next_ob, reward=reward))  # note that this doesn't take shift into account
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
        # # global reward
        agent_rollout_stats[-1] = AttrDict(
            total_reward=total_reward,
            steps=steps
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
                # idx=agent_idx[agent_id],
                steps=pos_rollout_stats[agent_id]['steps']+neg_rollout_stats[agent_id]['steps'])
            if agent_id > -1:
                # if it's not global, then record the index
                # print(combined_rollout_stats[agent_id])
                combined_rollout_stats[agent_id].idx = agent_idx[agent_id]

        # ok, so for global, you do not care about the idx, but you do care about the steps
        # what's the easiest way around this?

        return combined_rollout_stats


class ARS_Sampler(Base_ARS_Sampler):
    def __init__(self, 
            num_deltas, 
            shift,
            seed, 
            env_name,
            organism_builder,
            deltas_id,
            rollout_length,
            delta_std,
            num_workers,
            worker_builder=Worker):
        super(ARS_Sampler, self).__init__(
            num_deltas=num_deltas, 
            shift=shift,
            seed=seed, 
            env_name=env_name,
            organism_builder=organism_builder,
            deltas_id=deltas_id,
            rollout_length=rollout_length,
            delta_std=delta_std,
            num_workers=num_workers,
            worker_builder=worker_builder
            )

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
        global_rollout_rewards = []

        for result in results:
            for subresult in result:
                if not evaluate:
                    self.timesteps += subresult[-1]['steps']
                for agent_id in agent_ids:
                    if not evaluate:  #  this can be combined
                        agent_deltas_idx[agent_id].append(subresult[agent_id]['idx'])
                    else:
                        agent_deltas_idx[agent_id].append(-1)
                    agent_rollout_rewards[agent_id].append(subresult[agent_id]['total_reward'])
                global_rollout_rewards.append(subresult[-1]['total_reward'])

        for agent_id in agent_ids:
            agent_deltas_idx[agent_id] = np.array(agent_deltas_idx[agent_id])
            agent_rollout_rewards[agent_id] = np.array(agent_rollout_rewards[agent_id], dtype=np.float64)
        global_rollout_rewards = np.array(global_rollout_rewards, dtype=np.float64)
        agent_rollout_rewards[-1] = global_rollout_rewards  # global
        return agent_deltas_idx, agent_rollout_rewards

def create_auction(agent_args, auction_builder, agent_builder):
    import copy
    action_dim = agent_args['ac_dim']
    print('action_dim', action_dim)
    redundancy = 2
    subagent_args = copy.deepcopy(agent_args)
    subagent_args['ac_dim'] = 1
    agents = [agent_builder(subagent_args, id_num=i) for i in range(action_dim*redundancy)]
    return auction_builder(agents, action_dim)


class ARSExperiment(Base_ARSExperiment):
    """ 
    Object class implementing the ARS algorithm.
    """
    def __init__(self, 
                 organism_builder=None,
                 logdir=None, 
                 params=None,
                 master_organism=None,
                 sampler_builder=None,
                 ):

        super(ARSExperiment, self).__init__(
            organism_builder=organism_builder,
            logdir=logdir,
            params=params,
            master_organism=master_organism,
            sampler_builder=sampler_builder,
            )

    def eval_step(self, start, i):
        rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
        np.savez(self.logdir + "/lin_policy_plus", self.master_organism.get_state())
        
        print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        for agent_id in rewards:
            prefix = 'Global' if agent_id == -1 else 'Agent {}'.format(agent_id)
            logz.log_tabular("{} AverageReward".format(prefix), np.mean(rewards[agent_id]))
            logz.log_tabular("{} StdRewards".format(prefix), np.std(rewards[agent_id]))
            logz.log_tabular("{} MaxRewardRollout".format(prefix), np.max(rewards[agent_id]))
            logz.log_tabular("{} MinRewardRollout".format(prefix), np.min(rewards[agent_id]))

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
                     organism_builder=lambda: create_auction(
                        agent_args, ARS_LinearAuction, ARS_LinearAgent),
                     logdir=logdir,
                     params=params,
                     master_organism=create_auction(
                        agent_args, 
                        ARS_MasterLinearAuction, 
                        ARS_MasterLinearAgent),
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

