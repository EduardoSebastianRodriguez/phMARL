import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space
from functions import PIMARL_actor


class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        #############################################################################
        #############################################################################
        ######## TO USE THE PH-MARL POLICY PARAMETERIZATION, TURN THIS TRUE #########
        #############################################################################
        #############################################################################

        self.ph_MARL = True

        #############################################################################
        #############################################################################
        #############################################################################

        self.hidden_size = args.hidden_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        if self.ph_MARL:
            self.base = base(args, (7 + 1,))
            dummy = 1
        else:
            self.base = base(args, obs_shape)

        if self.ph_MARL:

            self.device= device

            self.num_agents = 6

            self.actor_config = {
                "device": device,
                "n_agents": self.num_agents,
                "observation_dim_per_agent": 63,
                "action_dim_per_agent": get_shape_from_obs_space(action_space)[0],
                "batch_size": 1000,
            }
            self.rnn = PIMARL_actor(self.actor_config)
        else:
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, dynamics, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.ph_MARL:
            agent_id = int(torch.argmax(obs[0, -6:]).detach().cpu().numpy())
            next_obs_after = torch.zeros(obs.shape[0], self.num_agents, 7, device=self.device)
            for agent in range(self.num_agents):
                next_obs_after[:, agent, 0:2] = obs[:, :2]
                next_obs_after[:, agent, 2:5] = obs[:, 8:11]
                next_obs_after[:, agent, 5] = obs[:, 2 + agent]
                next_obs_after[:, agent, 6] = obs[:, 11 + agent]
            actor_features = self.base(torch.cat((next_obs_after.reshape(-1, 7),
                                                  0.001 * torch.Tensor(dynamics).to(self.device).reshape(-1, 1)), dim=1)).reshape(obs.shape[0], self.num_agents, -1)
            # actor_features = next_obs_after
            actor_features = self.rnn(actor_features[:, :, :63], actor_features[:, :, 63:], agent_id)
        else:
            actor_features = self.base(obs)

            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, dynamics, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.ph_MARL:
            agent_id = int(torch.argmax(obs[0, -6:]).detach().cpu().numpy())
            next_obs_after = torch.zeros(obs.shape[0], self.num_agents, 7, device=self.device)
            for agent in range(self.num_agents):
                next_obs_after[:, agent, 0:2] = obs[:, :2]
                next_obs_after[:, agent, 2:5] = obs[:, 8:11]
                next_obs_after[:, agent, 5] = obs[:, 2 + agent]
                next_obs_after[:, agent, 6] = obs[:, 11 + agent]
            actor_features = self.base(torch.cat((next_obs_after.reshape(-1, 7),
                                                  0.001 * torch.Tensor(dynamics).to(self.device).reshape(-1, 1)),
                                                 dim=1)).reshape(obs.shape[0], self.num_agents, -1)
            # actor_features = next_obs_after
            actor_features = self.rnn(actor_features[:, :, :63], actor_features[:, :, 63:], agent_id)
        else:
            actor_features = self.base(obs)

            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.args.algorithm_name=="hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states