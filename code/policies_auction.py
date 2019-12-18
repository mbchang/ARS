from collections import OrderedDict
from operator import itemgetter

from policies import Base_ARS_Agent, ARS_MasterAgent


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__    

# slim version.
# will take in linear agents as input
class Base_ARS_Auction(Base_ARS_Agent):  # usually this subclasses nn.Module
    def __init__(self, agents):  # usually also have device and args
        super(Base_ARS_Auction, self).__init__()
        self.agents = agents  # usually this is a nn.ModuleList

    #########################################################
    # weight modifications

    def copy_weights(self):
        return {agent.id: agent.copy_weights() for agent in self.agents}

    def get_weight_shape(self):
        return {agent.id: agent.get_weight_shape() for agent in self.agents}

    def get_num_weights(self):
        return {agent.id: agent.get_num_weights() for agent in self.agents}

    def sync_weights(self, other):
        # note that the a_id should correspond
        for a_id, agent in enumerate(self.agents):
            assert a_id == agent.id == other.agents[a_id].id
            agent.sync_weights(other.agents[a_id])

    def add_noise_to_weights(self, noise_block):
        for a_id, agent in enumerate(self.agents):
            assert a_id == agent.id
            agent.add_noise_to_weights(noise_block[a_id])

    #########################################################
    # observation filter stuff

    def evaluate_mode(self):
        for agent in self.agents:
            agent.evaluate_mode()

    def train_mode(self):
        for agent in self.agents:
            agent.train_mode()

    def get_state(self):
        return {agent.id: agent.get_state() for agent in  self.agents}

    def stats_increment(self):
        for agent in self.agents:
            agent.stats_increment()
        return

    def get_filter(self):
        return {agent.id: agent.get_filter() for agent in self.agents}

    def sync_filter(self, other):
        for a_id,  agent in enumerate(self.agents):
            assert a_id == agent.id == other.agents[a_id].id
            agent.sync_filter(other.agents[a_id])
        return
    #########################################################


class ARS_MasterAuction(Base_ARS_Auction):
    def __init__(self, agents):
        Base_ARS_Auction.__init__(self, agents)

    #########################################################
    # observation filter stuff
    def update_filter(self, other):
        # note that the a_id should correspond
        for a_id, agent in enumerate(self.agents):
            assert a_id == agent.id == other.agents[a_id].id
            # print('updating filter for agent {}'.format(a_id))
            agent.update_filter(other.agents[a_id])

    def clear_filter_buffer(self):
        for agent in self.agents:
            agent.clear_filter_buffer()
    #########################################################


class ARS_LinearAuction(Base_ARS_Auction):
    def __init__(self, agents, action_dim):
        Base_ARS_Auction.__init__(self, agents)
        self.action_dim = action_dim
        self.bootstrap = True
        self.discrete = True
        self.args = AttrDict(gamma=1)

    def get_active_agents(self):
        active_agents = []
        for agent in self.agents:
            if agent.active:
                active_agents.append(agent)
        return active_agents

    def _run_auction(self, obs):
        assert len(self.get_active_agents()) == len(self.agents)  # for now let's not do dropout
        bids = OrderedDict([(a.id, a.forward(obs)) for a in self.get_active_agents()])
        return bids

    def _choose_winner(self, bids):
        winner = max(bids.items(), key=itemgetter(1))[0]
        return winner

    def _select_action(self, winner):
        return winner % self.action_dim

    def forward(self, obs):
        bids = self._run_auction(obs)
        winner = self._choose_winner(bids)
        action = self._select_action(winner)
        return action, winner, bids

    # bucket brigade
    def _compute_payoffs(self, state, bids, winner, next_winner_bid, next_state, reward):
        payoffs = {}
        for agent in self.get_active_agents():
            if agent.id == winner:
                revenue = reward + self.args.gamma*next_winner_bid
                payoffs[agent.id] = revenue - bids[winner]          
            else:
                payoffs[agent.id] = 0
        return payoffs


class ARS_MasterLinearAuction(ARS_LinearAuction, ARS_MasterAuction):
    def __init__(self, agents, action_dim):
        assert all(isinstance(agent, ARS_MasterAgent) for agent in agents)
        ARS_LinearAuction.__init__(self, agents, action_dim)  # only need to initialize once?

    def update(self, rl_alg, deltas_idx, rollout_rewards):
        # rollout_rewards.pop(-1)  #  pop the global reward here. Note that this mutates the object.

        #  you can just write it as you want it to look.
        assert len(self.get_active_agents()) == len(deltas_idx.keys()) == len(rollout_rewards.keys())-1
        for agent in self.get_active_agents():
            deltas_idx_agent = deltas_idx[agent.id]
            rollout_rewards_agent = rollout_rewards[agent.id]
            agent.update(rl_alg, deltas_idx_agent, rollout_rewards_agent)




