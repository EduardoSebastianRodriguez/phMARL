import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from reinforcement_functions import SquashedNormal


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MLP2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Attention_GNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim))
        self.Ak = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim))
        self.Av = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim))

        self.Bq = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim * self.na,
            [hidden_dim * self.na]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim * self.na,
            [output_dim * self.na]
        ).to(device)

    def forward(self, x):
        x = self.mlp_in(x.reshape(x.shape[0], -1)).reshape(x.shape[0], self.na, -1)

        Q = x
        K = x.transpose(1, 2)
        V = x

        output = torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V)

        return self.mlp_out(output.reshape(x.shape[0], -1)).reshape(output.shape[0], self.na, -1)


class Attention_Attention(nn.Module):
    '''
      Attention for R
    '''

    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        return self.mlp_out(x.mean(dim=1)).reshape(-1, self.na, self.output_dim)


class Attention_LEMURS(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        return self.mlp_out(x.mean(dim=1)).reshape(-1, self.na, self.output_dim)


class MLP_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.na = actor_config["n_agents"]
        self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.num_envs = actor_config["num_envs"]
        self.epsilon = 1e-6
        self.drag = 0.25  # From VMAS
        self.state_dim_per_agent = 4
        self.log_std_min = -5
        self.log_std_max = 2

        self.control_policy = MLP(
            in_channels=self.observation_dim_per_agent * self.na,
            hidden_channels=[self.action_dim_per_agent * self.na]
        ).to(self.device)

        self.std_net = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[self.action_dim_per_agent * self.na]
        ).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.contiguous(), torch.ones((1, self.na, 1), device=self.device))
        Q = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def forward(self, x):
        # Reshape input tensor
        x = x.reshape([x.shape[0], -1])

        # Copy input for later usage
        std_input = x.clone()

        # Batch size
        batch_size = x.shape[0]

        # Compute control policy
        u_mean = self.control_policy(x)
        u_log_std = self.std_net(torch.cat((std_input, u_mean), dim=1)).reshape(batch_size, self.na, -1)
        u_log_std = torch.tanh(u_log_std)
        u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        return SquashedNormal(u_mean.reshape(batch_size, self.na, -1), u_log_std.exp())


class MLP_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.mlp = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                         action.reshape([action.shape[0], -1])), dim=1))


class MSA_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.na = actor_config["n_agents"]
        self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.num_envs = actor_config["num_envs"]
        self.epsilon = 1e-6
        self.drag = 0.25  # From VMAS
        self.state_dim_per_agent = 4
        self.log_std_min = -5
        self.log_std_max = 2

        self.control_policy = Attention_GNN(self.observation_dim_per_agent,
                                        self.action_dim_per_agent,
                                        self.observation_dim_per_agent,
                                        self.na,
                                        self.device).to(self.device)
        self.std_net = Attention_GNN(self.observation_dim_per_agent + self.action_dim_per_agent,
                                 self.action_dim_per_agent,
                                 self.observation_dim_per_agent,
                                 self.na,
                                 self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.contiguous(), torch.ones((1, self.na, 1), device=self.device))
        Q = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def forward(self, x):

        # Copy input for later usage
        std_input = x.clone()

        u_mean = self.control_policy(x)
        u_log_std = self.std_net(torch.cat((std_input, u_mean), dim=2))
        u_log_std = torch.tanh(u_log_std)
        u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        return SquashedNormal(u_mean, u_log_std.exp())


class MSA_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.mlp = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                         action.reshape([action.shape[0], -1])), dim=1))


class GSA_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.na = actor_config["n_agents"]
        self.scenario_name = actor_config["scenario_name"]
        if self.scenario_name == "simple_spread":
            self.observation_dim_per_agent = 6
        else:
            self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.num_envs = actor_config["num_envs"]
        self.epsilon = 1e-6
        self.drag = 0.25  # From VMAS
        self.state_dim_per_agent = 4
        self.log_std_min = -5
        self.log_std_max = 2

        self.control_policy = Attention_Attention(self.observation_dim_per_agent,
                                        self.action_dim_per_agent,
                                        self.observation_dim_per_agent,
                                        self.na,
                                        self.device).to(self.device)
        self.std_net = Attention_Attention(self.observation_dim_per_agent + self.action_dim_per_agent,
                                 self.action_dim_per_agent,
                                 self.observation_dim_per_agent,
                                 self.na,
                                 self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.contiguous(), torch.ones((1, self.na, 1), device=self.device))
        Q = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def forward(self, x):
        if self.scenario_name == "simple_spread":
            state = torch.zeros(x.shape[0], x.shape[1], self.observation_dim_per_agent, device=self.device)
            state[:, :, 0:4] = torch.clone(x[:, :, 0:4])
            for i in range(self.na):
                state[:, i, 4:] = torch.clone(x[:, i, 4 + 2*i:6 + 2*i])
        else:
            state = x

        # Laplacian
        laplacian = self.laplacian(state[:, :, 0:2])
        laplacian = torch.kron(laplacian, torch.ones((1, 1, self.observation_dim_per_agent), device=self.device))
        laplacian = laplacian.reshape(-1, self.na, self.observation_dim_per_agent)

        # Reshape and normalize inputs
        state = state.repeat(1, self.na, 1)
        state = state.reshape(-1, self.na, self.observation_dim_per_agent)
        state = (laplacian * state)

        # Copy input for later usage
        std_input = state.clone()

        u_mean = self.control_policy(state)
        u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, u_mean.shape[2]).unsqueeze(1).repeat(1, self.na, 1)), dim=2))
        u_log_std = torch.tanh(u_log_std)
        u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        return SquashedNormal(u_mean, u_log_std.exp())


class GSA_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.mlp = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                         action.reshape([action.shape[0], -1])), dim=1))


class LEMURS_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.num_envs = actor_config["num_envs"]
        self.na = actor_config["n_agents"]
        self.scenario_name = actor_config["scenario_name"]
        if self.scenario_name == "simple_spread":
            self.observation_dim_per_agent = 6
        else:
            self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.epsilon = 1e-6
        self.drag = 0.25  # From VMAS
        self.state_dim_per_agent = 4
        self.log_std_min = -5
        self.log_std_max = 2

        self.R_mean = Att_R(self.observation_dim_per_agent,
                            16,
                            8,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent,
                            16,
                            8,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)
        self.H_mean = Att_H(self.observation_dim_per_agent,
                            25,
                            8,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)

        self.std_net = Attention_LEMURS(self.observation_dim_per_agent + self.action_dim_per_agent,
                                 self.action_dim_per_agent,
                                 self.observation_dim_per_agent,
                                 self.na,
                                 self.device).to(self.device)


    def laplacian(self, q_agents):
        Q1 = q_agents.repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.contiguous(), torch.ones((1, self.na, 1), device=self.device))
        Q = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def forward(self, x):

        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                 self.action_dim_per_agent * self.na,
                                                 device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1)

        self.J_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1),
                                torch.cat((-torch.eye(self.action_dim_per_agent * self.na, device=self.device),
                                torch.zeros(self.action_dim_per_agent * self.na,
                                            self.action_dim_per_agent * self.na, device=self.device)), dim=1)
                                ), dim=0)
        self.R_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                 torch.zeros(self.action_dim_per_agent * self.na,
                                             self.action_dim_per_agent * self.na,
                                             device=self.device)), dim=1),
                                torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                self.drag*torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1)
                                ), dim=0)

        batch_size = x.shape[0]
        if self.scenario_name == "simple_spread":
            state = torch.zeros(x.shape[0], x.shape[1], self.observation_dim_per_agent, device=self.device)
            state[:, :, 0:4] = torch.clone(x[:, :, 0:4])
            for i in range(self.na):
                state[:, i, 4:] = torch.clone(x[:, i, 4 + 2*i:6 + 2*i])
        else:
            state = x
        state_h_mean = torch.clone(state).reshape(-1, self.observation_dim_per_agent)

        # Laplacian
        laplacian_base = self.laplacian(state[:, :, 0:2])
        laplacian = torch.kron(laplacian_base, torch.ones((1, 1, self.observation_dim_per_agent), device=self.device))
        laplacian = laplacian.reshape(-1, self.na, self.observation_dim_per_agent)

        # Reshape and normalize inputs
        state = state.repeat(1, self.na, 1)
        state = state.reshape(-1, self.na, self.observation_dim_per_agent)
        state = (laplacian * state)

        # Copy input for later usage
        std_input = state.clone()

        R_mean = self.R_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.na)
            Hgrad_mean = torch.autograd.grad(H_mean.sum(), state_h_mean, only_inputs=True, create_graph=True)
            dH_mean = Hgrad_mean[0]
        dHq_mean = dH_mean[:, :self.action_dim_per_agent].reshape(-1,
                                                                   self.na * self.action_dim_per_agent)
        dHp_mean = dH_mean[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
                                                                     self.na * self.action_dim_per_agent)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)

        # Controller dynamics
        F_sys_pinv = self.F_sys_pinv.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        R_sys = self.R_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        J_sys = self.J_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
                                   dx_mean[:, :self.action_dim_per_agent * self.na].unsqueeze(dim=2)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.na, -1)

        u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, u_mean.shape[2]).unsqueeze(1).repeat(1, self.na, 1)), dim=2))
        u_log_std = torch.tanh(u_log_std)
        u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        return SquashedNormal(u_mean, u_log_std.exp())


class LEMURS_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.scenario_name = qvalue_config["scenario_name"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.mlp = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                   action.reshape([action.shape[0], -1])), dim=1))


class Attention_preprocessor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP2(
            input_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, laplacian):
        self.na = x.shape[1]

        x = (self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1) *
             torch.kron(laplacian, torch.ones((1, 1, self.hidden_dim), device=self.device))
             .reshape(x.shape[0], self.na, -1))

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = (self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim) *
             torch.kron(laplacian, torch.ones((1, 1, self.output_dim), device=self.device))
             .reshape(x.shape[0], self.na, -1)).mean(dim=1).reshape(-1, self.na, self.output_dim)
        return x


class PIMARL_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.num_envs = actor_config["num_envs"]
        self.scenario_name = actor_config["scenario_name"]
        self.na = actor_config["n_agents"]
        if self.scenario_name == "food_collection_vmas" or (self.scenario_name == "simple_spread_food" and not actor_config["preprocessor"]) or self.scenario_name == "simple_spread":
            self.observation_dim_per_agent = 6
        elif self.scenario_name == "grassland_vmas" or self.scenario_name == "adversarial_vmas":
            if actor_config["preprocessor"]:
                self.observation_dim_per_agent = 10
            else:
                self.observation_dim_per_agent = 8
        elif self.scenario_name == "simple_spread_food" and actor_config["preprocessor"]:
            self.observation_dim_per_agent = 8
        else:
            self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.ratio = actor_config["ratio"]
        self.ratio_eval = actor_config["ratio_eval"]
        self.epsilon = 1e-6
        self.drag = 0.25  # From VMAS
        self.state_dim_per_agent = 4
        self.log_std_min = -5
        self.log_std_max = 2
        self.R_mean = Att_R(self.observation_dim_per_agent,
                            16,
                            8,
                            self.observation_dim_per_agent,
                            self.scenario_name,
                            self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent,
                            16,
                            8,
                            self.observation_dim_per_agent,
                            self.scenario_name,
                            self.device).to(self.device)
        self.H_mean = Att_H(self.observation_dim_per_agent,
                            25,
                            8,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)

        self.std_net = Attention_LEMURS(self.observation_dim_per_agent + self.action_dim_per_agent,
                                 self.action_dim_per_agent,
                                 self.observation_dim_per_agent,
                                 self.na,
                                 self.device).to(self.device)

        self.preprocessor = Attention_preprocessor(self.observation_dim_per_agent - 2,
                                             2,
                                             2,
                                             self.na,
                                             self.device).to(self.device)
        self.prepro = actor_config["preprocessor"]
        self.u_mean_prev = 0
        self.u_log_std_prev = 0


    def laplacian(self, q_agents):
        Q1 = q_agents.repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.contiguous(), torch.ones((1, self.na, 1), device=self.device))
        Q = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, q_agents.shape[1])
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def forward(self, x):

        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                 self.action_dim_per_agent * self.na,
                                                 device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1)

        self.J_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1),
                                torch.cat((-torch.eye(self.action_dim_per_agent * self.na, device=self.device),
                                torch.zeros(self.action_dim_per_agent * self.na,
                                            self.action_dim_per_agent * self.na, device=self.device)), dim=1)
                                ), dim=0)
        self.R_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                 torch.zeros(self.action_dim_per_agent * self.na,
                                             self.action_dim_per_agent * self.na,
                                             device=self.device)), dim=1),
                                torch.cat((torch.zeros(self.action_dim_per_agent * self.na,
                                                       self.action_dim_per_agent * self.na,
                                                       device=self.device),
                                self.drag*torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1)
                                ), dim=0)

        batch_size = x.shape[0]

        # Laplacian
        laplacian_base = self.laplacian(x[:, :, 0:2])
        laplacian = torch.kron(laplacian_base, torch.ones((1, 1, self.observation_dim_per_agent), device=self.device))
        laplacian = laplacian.reshape(-1, self.na, self.observation_dim_per_agent)

        if self.scenario_name == "simple_spread_food" and self.prepro:
            state = torch.cat((x, self.preprocessor(x.repeat(1, self.na, 1).reshape(-1, self.na, 6), laplacian_base)), dim=2)
        elif (self.scenario_name == "grassland_vmas" or self.scenario_name == "adversarial_vmas") and self.prepro:
            state = torch.cat((x, self.preprocessor(x.repeat(1, self.na, 1).reshape(-1, self.na, 8), laplacian_base)), dim=2)
        elif self.scenario_name == "simple_spread":
            state = torch.zeros(x.shape[0], x.shape[1], self.observation_dim_per_agent, device=self.device)
            state[:, :, 0:4] = torch.clone(x[:, :, 0:4])
            state[:, :, 4:6] = torch.clone(x).reshape(x.shape[0], x.shape[1], -1, 2)[:, range(self.na), range(2, 2+self.na), :].reshape(x.shape[0], x.shape[1], -1)
        else:
            state = x
        state_h_mean = torch.clone(state).reshape(-1, self.observation_dim_per_agent)

        # Reshape and normalize inputs
        if self.scenario_name == "simple_spread_food":
            state[:, :, 0:2] *= self.ratio/self.ratio_eval
            state[:, :, 4:6] *= self.ratio/self.ratio_eval
            state_h_mean[:, 0:2] *= self.ratio/self.ratio_eval
            state_h_mean[:, 4:6] *= self.ratio/self.ratio_eval


        state = state.repeat(1, self.na, 1)
        state = state.reshape(-1, self.na, self.observation_dim_per_agent)
        state = (laplacian * state)

        # Copy input for later usage
        std_input = state.clone()

        R_mean = self.R_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.na)
            Hgrad_mean = torch.autograd.grad(H_mean.sum(), state_h_mean, only_inputs=True, create_graph=True)
            dH_mean = Hgrad_mean[0]
        dHq_mean = dH_mean[:, :self.action_dim_per_agent].reshape(-1,
                                                                   self.na * self.action_dim_per_agent)
        dHp_mean = dH_mean[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
                                                                     self.na * self.action_dim_per_agent)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)

        # Controller dynamics
        F_sys_pinv = self.F_sys_pinv.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        R_sys = self.R_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        J_sys = self.J_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
                                   dx_mean[:, :self.action_dim_per_agent * self.na].unsqueeze(dim=2)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.na, -1)

        u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, u_mean.shape[2]).unsqueeze(1).repeat(1, self.na, 1)), dim=2))
        u_log_std = torch.tanh(u_log_std)
        u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)
        # if torch.isnan(u_mean).any() or torch.isnan(u_log_std).any():
        #     u_mean = 2.0 * (torch.rand(u_mean.shape, device=self.device) - 0.5)
        #     u_log_std = 2.0 * (torch.rand(u_log_std.shape, device=self.device) - 0.5)
        if self.scenario_name == "grassland_vmas" or self.scenario_name == "adversarial_vmas":
            return SquashedNormal(u_mean[:, int(self.na/2):, :], u_log_std[:, int(self.na/2):, :].exp())
        else:
            return SquashedNormal(u_mean, u_log_std.exp())


class PIMARL_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.scenario_name = qvalue_config["scenario_name"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        if self.scenario_name == "grassland_vmas" or self.scenario_name == "adversarial_vmas":
            self.mlp = MLP(
                in_channels=self.observation_dim_per_agent * self.na + self.action_dim_per_agent * int(self.na/2),
                hidden_channels=[self.observation_dim_per_agent * self.na * 2 + self.action_dim_per_agent * 2 * int(self.na/2),
                                 self.observation_dim_per_agent * self.na + self.action_dim_per_agent * int(self.na/2),
                                 self.observation_dim_per_agent + self.action_dim_per_agent,
                                 1]
            ).to(self.device)
        else:
            self.mlp = MLP(
                in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                                 (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                                 (self.observation_dim_per_agent + self.action_dim_per_agent),
                                 1]
            ).to(self.device)

    def forward(self, observation, action):
        if self.scenario_name == "grassland_vmas" or self.scenario_name == "adversarial_vmas":
            return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                       action[:, int(self.na/2):, :].reshape([action.shape[0], -1])), dim=1))
        else:
            return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]),
                                       action.reshape([action.shape[0], -1])), dim=1))


class Att_R(nn.Module):
    '''
      Attention for R
    '''

    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()

        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]

        if scenario_name == "simple_spread_food" or scenario_name == "grassland_vmas" or scenario_name == "adversarial_vmas":
            x = (self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1) *
                 torch.kron(laplacian, torch.ones((1, 1, 2 * self.hidden_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1))

            Q = self.activation_swish(
                torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
            K = self.activation_swish(
                torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

            x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = (self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1) *
                 torch.kron(laplacian, torch.ones((1, 1, self.hidden_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1))

            Q = self.activation_swish(
                torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
            K = self.activation_swish(
                torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

            x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = (self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim) *
                 torch.kron(laplacian, torch.ones((1, 1, self.output_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1)).transpose(1, 2)
        else:
            x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

            Q = self.activation_swish(
                torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
            K = self.activation_swish(
                torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

            x = self.activation_swish(
                torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

            Q = self.activation_swish(
                torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
            K = self.activation_swish(
                torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

            x = self.activation_swish(
                torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        R11 = x[:, 0:4, :].sum(1)
        R12 = x[:, 4:8, :].sum(1)
        R21 = x[:, 8:12, :].sum(1)
        R22 = x[:, 12:16, :].sum(1)
        R11 = R11.reshape(batch, self.na, self.na)
        R12 = R12.reshape(batch, self.na, self.na)
        R21 = R21.reshape(batch, self.na, self.na)
        R22 = R22.reshape(batch, self.na, self.na)
        R = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        R = R ** 2

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag = torch.clone(Rupper)
        Rdiag[:, range(self.na), range(self.na)] = torch.zeros(self.na, device=self.device)
        Rout = torch.eye(2 * self.na, device=self.device).unsqueeze(dim=0).repeat(batch, 1, 1) * Rupper
        R = Rout + torch.eye(2 * self.na, device=self.device).unsqueeze(dim=0).repeat(batch, 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return torch.kron(R, torch.eye(2, device=self.device).unsqueeze(0))


class Att_J(nn.Module):
    '''
      Attention for J
    '''

    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()

        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]

        if scenario_name == "simple_spread_food" or scenario_name == "grassland_vmas" or scenario_name == "adversarial_vmas":
            x = (self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1) *
                 torch.kron(laplacian, torch.ones((1, 1, 2 * self.hidden_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1))

            Q = self.activation_swish(
                torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
            K = self.activation_swish(
                torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

            x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = (self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1) *
                 torch.kron(laplacian, torch.ones((1, 1, self.hidden_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1))

            Q = self.activation_swish(
                torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
            K = self.activation_swish(
                torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

            x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = (self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim) *
                 torch.kron(laplacian, torch.ones((1, 1, self.output_dim), device=self.device))
                 .reshape(x.shape[0], self.na, -1)).transpose(1, 2)
        else:
            x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

            Q = self.activation_swish(
                torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
            K = self.activation_swish(
                torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

            x = self.activation_swish(
                torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

            Q = self.activation_swish(
                torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
            K = self.activation_swish(
                torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
            V = self.activation_swish(
                torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

            x = self.activation_swish(
                torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

            x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        # Reshape, kronecker and post-processing to ensure skew-symmetry
        batch = int(x.shape[0] / x.shape[2])

        J11 = torch.zeros((batch, self.na, self.na), device=self.device)
        J22 = torch.zeros((batch, self.na, self.na), device=self.device)
        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -torch.clone(j12)
        J12 = torch.zeros((batch, self.na, self.na), device=self.device)
        J21 = torch.zeros((batch, self.na, self.na), device=self.device)
        J12[:, range(self.na), range(self.na)] = j12
        J21[:, range(self.na), range(self.na)] = j21
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)

        return torch.kron(J, torch.eye(2, device=self.device).unsqueeze(0))


class Att_H(nn.Module):
    '''
      Attention for H
    '''

    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, na):
        self.na = na
        x = self.mlp_in(x).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).unsqueeze(dim=1).transpose(1, 2)

        # Reshape, kronecker and post-processing
        l = 2
        M11 = torch.kron((x[:, 0:5, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((x[:, 5:10, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((x[:, 10:15, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((x[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (x[:, 20:25, :] ** 2).sum(1)
        Mupper11 = torch.zeros([x.shape[0], l, l], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l, l], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l, l], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l, l], device=self.device)
        Mupper11[:, range(l), range(l)] = M11
        Mupper12[:, range(l), range(l)] = M12
        Mupper21[:, range(l), range(l)] = M21
        Mupper22[:, range(l), range(l)] = M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)
        q = x[:, :4, :]

        return torch.bmm(q.transpose(1, 2), torch.bmm(M, q)).sum(2) + Mpp.sum(1).unsqueeze(1)


class MADDPG(nn.Module):
    def __init__(self, in_channels, out_channels, agent_index, device):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.agent_index = agent_index
        self.device = device
        self.radius = 0.15
        self.result_prev = 0.

        layers = [nn.Linear(self.in_channels, 64),
                  nn.ReLU(),
                  nn.Linear(64, 64),
                  nn.ReLU(),
                  nn.Linear(64, self.out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        observation = torch.zeros(x.shape[0], self.in_channels, device=self.device)
        observation[:, 0:5] = x[:, self.agent_index, 0:5]

        index = np.random.randint(int(x.shape[1]/2))
        observation[:, 5:7] = x[:, index, 5:7] + x[:, index, 0:2] - x[:, self.agent_index, 0:2]
        observation[:, 7] = 0
        index = np.random.randint(int(x.shape[1]/2))
        observation[:, 8:10] = x[:, index, 5:7] + x[:, index, 0:2] - x[:, self.agent_index, 0:2]
        observation[:, 10] = 0
        index = np.random.randint(int(x.shape[1]/2))
        observation[:, 11:13] = x[:, index, 5:7] + x[:, index, 0:2] - x[:, self.agent_index, 0:2]
        observation[:, 13] = 0
        index = int(x.shape[1]/2) + np.random.randint(int(x.shape[1]/2))
        dist1 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 2)
        dist2 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 3)
        observation[:, 14:16][dist1 < self.radius] = (x[:, index, :2] - x[:, self.agent_index, 0:2])[dist1 < self.radius]
        observation[:, 16:19][dist2 < self.radius] = x[:, index, 2:5][dist2 < self.radius]
        index = int(x.shape[1]/2) + np.random.randint(int(x.shape[1]/2))
        dist1 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 2)
        dist2 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 3)
        observation[:, 19:21][dist1 < self.radius] = (x[:, index, :2] - x[:, self.agent_index, 0:2])[dist1 < self.radius]
        observation[:, 21:24][dist2 < self.radius] = x[:, index, 2:5][dist2 < self.radius]
        index = int(x.shape[1]/2) + np.random.randint(int(x.shape[1]/2))
        dist1 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 2)
        dist2 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 3)
        observation[:, 24:26][dist1 < self.radius] = (x[:, index, :2] - x[:, self.agent_index, 0:2])[dist1 < self.radius]
        observation[:, 26:29][dist2 < self.radius] = x[:, index, 2:5][dist2 < self.radius]
        index = int(x.shape[1]/2) + np.random.randint(int(x.shape[1]/2))
        dist1 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 2)
        dist2 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 3)
        observation[:, 29:31][dist1 < self.radius] = (x[:, index, :2] - x[:, self.agent_index, 0:2])[dist1 < self.radius]
        observation[:, 31:34][dist2 < self.radius] = x[:, index, 2:5][dist2 < self.radius]
        index = int(x.shape[1]/2) + np.random.randint(int(x.shape[1]/2))
        dist1 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 2)
        dist2 = torch.linalg.norm(x[:, index, :2] - x[:, self.agent_index, 0:2], dim=1).unsqueeze(dim=1).repeat(1, 3)
        observation[:, 34:36][dist1 < self.radius] = (x[:, index, :2] - x[:, self.agent_index, 0:2])[dist1 < self.radius]
        observation[:, 36:39][dist2 < self.radius] = x[:, index, 2:5][dist2 < self.radius]

        # Depending on the scenario, the parameters of the l1 and l2 change
        # Grassland: l1=0.1, l2=2
        # Adversarial: l1=0.1, l2=2
        l1 = 0.1
        l2 = 2.0

        result = self.layers(observation)
        result = torch.cat(((result[:, 1]-result[:, 2]).unsqueeze(1), (result[:, 3]-result[:, 4]).unsqueeze(1)), dim=1) * l1
        result[torch.abs(result) > 1.0] = torch.sign(result[torch.abs(result) > 1]) * l2

        return result


