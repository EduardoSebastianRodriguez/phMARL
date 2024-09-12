import torch
from torch import nn
from torch.autograd import Variable
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


class PIMARL_actor(nn.Module):

    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.na = actor_config["n_agents"]
        self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.batch_size = actor_config["batch_size"]
        self.epsilon = 1e-6
        self.log_std_min = -5
        self.log_std_max = 2
        self.R_mean = Att_R(self.observation_dim_per_agent,
                            16,
                            2,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent,
                            16,
                            2,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)
        self.H_mean = Att_H(self.observation_dim_per_agent,
                            25,
                            2,
                            self.observation_dim_per_agent,
                            self.device).to(self.device)

        # self.std_net = Attention_LEMURS(self.observation_dim_per_agent + self.action_dim_per_agent,
        #                          self.action_dim_per_agent,
        #                          self.observation_dim_per_agent,
        #                          self.na,
        #                          self.device).to(self.device)

        # self.std_net = 0.1 * torch.ones(1, self.na, self.action_dim_per_agent, device=self.device)

        # self.std_net = MLP(self.observation_dim_per_agent + self.action_dim_per_agent,
        #                    [20, 20, self.action_dim_per_agent]).to(self.device)

        self.ppo = True
        self.det = True

        if self.ppo:
            self.MLP_1 = MLP(self.action_dim_per_agent + self.observation_dim_per_agent,
                             [256, 256, 64]).to(self.device)
            self.MLP_2 = MLP(self.action_dim_per_agent + self.observation_dim_per_agent,
                             [256, 256, 64]).to(self.device)
            self.MLP_3 = MLP(self.action_dim_per_agent + self.observation_dim_per_agent,
                             [256, 256, 64]).to(self.device)
        else:
            self.MLP_1 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)
            self.MLP_2 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)
            self.MLP_3 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)
            self.MLP_4 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)
            self.MLP_5 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)
            self.MLP_6 = MLP(self.action_dim_per_agent + 3 * self.observation_dim_per_agent,
                             [256, 256, self.action_dim_per_agent]).to(self.device)

        # self.scale = nn.Parameter(torch.Tensor([0.001])).to(self.device)
        # self.scale = 0.001
        # self.gain = 0.001
        #
        # self.u_mean_prev = 0
        # self.u_log_std_prev = 0

    def laplacian(self):
        return torch.as_tensor([[1., 1., 0., 0., 0., 1.],
                                     [1., 1., 1., 0., 0., 0.],
                                     [0., 1., 1., 1., 0., 0.],
                                     [0., 0., 1., 1., 1., 0.],
                                     [0., 0., 0., 1., 1., 1.],
                                     [1., 0., 0., 0., 1., 1.]], device=self.device)

    def forward(self, x, dynamics, agent_id):

        # if len(dynamics.shape) == 3:
        #     dynamics = dynamics[:, 0, :]

        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * self.na, self.action_dim_per_agent * self.na, device=self.device),
                                     torch.eye(self.action_dim_per_agent * self.na, device=self.device)), dim=1)

        batch_size = x.shape[0]

        # Laplacian
        laplacian_base = self.laplacian().unsqueeze(0).repeat(batch_size, 1, 1)
        laplacian = torch.kron(laplacian_base, torch.ones((1, 1, self.observation_dim_per_agent), device=self.device))
        laplacian = laplacian.reshape(-1, self.na, self.observation_dim_per_agent)

        state = x
        state_h_mean = torch.clone(state).reshape(-1, self.observation_dim_per_agent)
        state = state.repeat(1, self.na, 1)
        state = state.reshape(-1, self.na, self.observation_dim_per_agent)
        state = (laplacian * state)

        # Copy input for later usage
        # std_input = state.clone()

        R_mean = self.R_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32))
        J_mean = self.J_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32))
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.na)
            Hgrad_mean = torch.autograd.grad(H_mean.sum(), state_h_mean, only_inputs=True, create_graph=True)
            dH_mean = Hgrad_mean[0].reshape(batch_size, self.na, -1)
        # dH_mean_1 = dH_mean[:, 0, 5:7]
        # dH_mean_2 = dH_mean[:, 1, 5:7]
        # dH_mean_3 = dH_mean[:, 2, 5:7]
        # dH_mean_4 = dH_mean[:, 3, 5:7]
        # dH_mean_5 = dH_mean[:, 0, 13:15]
        # dH_mean_6 = dH_mean[:, 1, 13:15]
        # dH_mean_7 = dH_mean[:, 2, 13:15]
        # dH_mean_8 = dH_mean[:, 3, 13:15]
        dH_mean_1 = dH_mean[:, 0, 5].reshape(batch_size, 1)
        dH_mean_2 = dH_mean[:, 1, 5].reshape(batch_size, 1)
        dH_mean_3 = dH_mean[:, 2, 5].reshape(batch_size, 1)
        dH_mean_4 = dH_mean[:, 3, 5].reshape(batch_size, 1)
        dH_mean_5 = dH_mean[:, 4, 5].reshape(batch_size, 1)
        dH_mean_6 = dH_mean[:, 5, 5].reshape(batch_size, 1)
        dH_mean_7 = dH_mean[:, 0, 6].reshape(batch_size, 1)
        dH_mean_8 = dH_mean[:, 1, 6].reshape(batch_size, 1)
        dH_mean_9 = dH_mean[:, 2, 6].reshape(batch_size, 1)
        dH_mean_10 = dH_mean[:, 3, 6].reshape(batch_size, 1)
        dH_mean_11 = dH_mean[:, 4, 6].reshape(batch_size, 1)
        dH_mean_12 = dH_mean[:, 5, 6].reshape(batch_size, 1)

        # dHq_mean = dH_mean[:, :self.action_dim_per_agent].reshape(-1,
        #                                                            self.na * self.action_dim_per_agent)
        # dHp_mean = dH_mean[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
        #                                                              self.na * self.action_dim_per_agent)
        # dHdx_mean = torch.cat((dH_mean_1, dH_mean_2, dH_mean_3, dH_mean_4,
        #                        dH_mean_5, dH_mean_6, dH_mean_7, dH_mean_8), dim=1)
        dHdx_mean = torch.cat((dH_mean_1, dH_mean_2, dH_mean_3, dH_mean_4, dH_mean_5, dH_mean_6,
                               dH_mean_7, dH_mean_8, dH_mean_9, dH_mean_10, dH_mean_11, dH_mean_12), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2).to(torch.float32)).squeeze(2)

        # Controller dynamics
        F_sys_pinv = self.F_sys_pinv.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        # new_dynamics = torch.cat((x[:, 0, 13:15].to(torch.float32),
        #                           x[:, 1, 13:15].to(torch.float32),
        #                           x[:, 2, 13:15].to(torch.float32),
        #                           x[:, 3, 13:15].to(torch.float32),
        #                           self.scale * dynamics[:, 6:].to(torch.float32)), dim=1)

        if self.det:
            new_dynamics = torch.cat((x[:, 0, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 1, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 2, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 3, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 4, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 5, 6].unsqueeze(1).to(torch.float32),
                                      dynamics.squeeze(2).to(torch.float32)), dim=1)
        else:
            new_dynamics = torch.cat((x[:, 0, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 1, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 2, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 3, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 4, 6].unsqueeze(1).to(torch.float32),
                                      x[:, 5, 6].unsqueeze(1).to(torch.float32),
                                      self.scale * dynamics.to(torch.float32)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - new_dynamics.unsqueeze(dim=2)).squeeze(dim=2).reshape(batch_size, self.na, -1)

        if self.det:
            u_mean_1 = self.MLP_1(torch.cat((u_mean[:, agent_id, :], x[:, agent_id, :].to(torch.float32)), dim=1))
            u_mean_2 = self.MLP_2(torch.cat((u_mean[:, agent_id, :], x[:, (agent_id + 1) % self.na, :].to(torch.float32)), dim=1))
            u_mean_3 = self.MLP_3(torch.cat((u_mean[:, agent_id, :], x[:, (agent_id - 1) % self.na, :].to(torch.float32)), dim=1))
        else:
            u_mean_1 = self.MLP_1(torch.cat((u_mean[:, 0, :], x[:, 0, :].to(torch.float32), x[:, 1, :].to(torch.float32), x[:, 5, :].to(torch.float32)), dim=1)).unsqueeze(1)
            u_mean_2 = self.MLP_2(torch.cat((u_mean[:, 1, :], x[:, 1, :].to(torch.float32), x[:, 2, :].to(torch.float32), x[:, 0, :].to(torch.float32)), dim=1)).unsqueeze(1)
            u_mean_3 = self.MLP_3(torch.cat((u_mean[:, 2, :], x[:, 2, :].to(torch.float32), x[:, 3, :].to(torch.float32), x[:, 1, :].to(torch.float32)), dim=1)).unsqueeze(1)
            u_mean_4 = self.MLP_4(torch.cat((u_mean[:, 3, :], x[:, 3, :].to(torch.float32), x[:, 4, :].to(torch.float32), x[:, 2, :].to(torch.float32)), dim=1)).unsqueeze(1)
            u_mean_5 = self.MLP_5(torch.cat((u_mean[:, 4, :], x[:, 4, :].to(torch.float32), x[:, 5, :].to(torch.float32), x[:, 3, :].to(torch.float32)), dim=1)).unsqueeze(1)
            u_mean_6 = self.MLP_6(torch.cat((u_mean[:, 5, :], x[:, 5, :].to(torch.float32), x[:, 0, :].to(torch.float32), x[:, 4, :].to(torch.float32)), dim=1)).unsqueeze(1)

        if self.det:
            u_mean = (u_mean_1 + u_mean_2 + u_mean_3) / 3
        else:
            u_mean = torch.cat((u_mean_1, u_mean_2, u_mean_3, u_mean_4, u_mean_5, u_mean_6), dim=1)

        # u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, u_mean.shape[2]).unsqueeze(1).repeat(1, self.na, 1)), dim=2))
        # u_log_std = torch.tanh(u_log_std)
        # u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        # u_log_std = self.std_net(torch.cat((x.reshape(-1, self.observation_dim_per_agent),
        #                                     u_mean.reshape(-1, self.action_dim_per_agent)), dim=1).to(torch.float32))
        # u_log_std = torch.tanh(u_log_std.reshape(batch_size, self.na, self.action_dim_per_agent))
        # u_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (u_log_std + 1)

        # return SquashedNormal(u_mean, u_log_std.exp())

        if self.det:
            return u_mean
        else:
            return SquashedNormal(u_mean, self.std_net.repeat(u_mean.shape[0], 1, 1))


class PIMARL_qvalue(nn.Module):

    def __init__(self, qvalue_config):
        super().__init__()

        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.mlp = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[256,
                             256,
                             1]
        ).to(self.device)

    def forward(self, observation, action):
        return self.mlp(torch.cat((observation.reshape([observation.shape[0], -1]), action.reshape([action.shape[0], -1])), dim=1))


class Att_R(nn.Module):
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

        # self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        # self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        # self.mlp_hidden_4 = MLP(
        #     2 * hidden_dim,
        #     [hidden_dim]
        # ).to(device)

        self.mlp_out = MLP(
            2 * hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, laplacian):
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        # x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        # Q = self.activation_swish(
        #     torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(
        #         dim=0).repeat(x.shape[0], 1, 1))
        # K = self.activation_swish(
        #     torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(
        #         dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        # V = self.activation_swish(
        #     torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(
        #        dim=0).repeat(x.shape[0], 1, 1))

        # x = self.activation_swish(
        #     torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, 2 * self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        # batch = int(x.shape[0] / x.shape[1])

        # batch = 1 if batch == 0 else batch
        #
        # x = x.reshape(batch, self.na, self.na, self.output_dim)

        # R = torch.zeros(batch, self.na * self.na, self.output_dim, device=self.device)
        # R[:, 0, :] = x[:, 0, 0, :]
        # R[:, 1, :] = x[:, 0, 1, :]
        # R[:, 8, :] = x[:, 0, 2, :]
        # R[:, 9, :] = x[:, 0, 3, :]
        #
        # R[:, 2, :] = x[:, 1, 0, :]
        # R[:, 3, :] = x[:, 1, 1, :]
        # R[:, 10, :] = x[:, 1, 2, :]
        # R[:, 11, :] = x[:, 1, 3, :]
        #
        # R[:, 4, :] = x[:, 2, 0, :]
        # R[:, 5, :] = x[:, 2, 1, :]
        # R[:, 12, :] = x[:, 2, 2, :]
        # R[:, 13, :] = x[:, 2, 3, :]
        #
        # R[:, 6, :] = x[:, 3, 0, :]
        # R[:, 7, :] = x[:, 3, 1, :]
        # R[:, 14, :] = x[:, 3, 2, :]
        # R[:, 15, :] = x[:, 3, 3, :]

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

        return torch.bmm(R, R.transpose(1, 2))


class Att_J(nn.Module):
    '''
      Attention for J
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

        # self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        # self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        # self.mlp_hidden_4 = MLP(
        #     2 * hidden_dim,
        #     [hidden_dim]
        # ).to(device)

        self.mlp_out = MLP(
            2 * hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x, laplacian):
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(
                dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        # x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        # Q = self.activation_swish(
        #     torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(
        #         dim=0).repeat(x.shape[0], 1, 1))
        # K = self.activation_swish(
        #     torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(
        #         dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        # V = self.activation_swish(
        #     torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(
        #         dim=0).repeat(x.shape[0], 1, 1))

        # x = self.activation_swish(
        #     torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, 2 * self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        # batch = int(x.shape[0] / x.shape[1])

        # batch = 1 if batch == 0 else batch
        #
        # x = x.reshape(batch, self.na, self.na, self.output_dim)
        #
        # J = torch.zeros(batch, self.na * self.na, self.output_dim, device=self.device)
        # J[:, 0, :] = x[:, 0, 0, :]
        # J[:, 1, :] = x[:, 0, 1, :]
        # J[:, 8, :] = x[:, 0, 2, :]
        # J[:, 9, :] = x[:, 0, 3, :]
        #
        # J[:, 2, :] = x[:, 1, 0, :]
        # J[:, 3, :] = x[:, 1, 1, :]
        # J[:, 10, :] = x[:, 1, 2, :]
        # J[:, 11, :] = x[:, 1, 3, :]
        #
        # J[:, 4, :] = x[:, 2, 0, :]
        # J[:, 5, :] = x[:, 2, 1, :]
        # J[:, 12, :] = x[:, 2, 2, :]
        # J[:, 13, :] = x[:, 2, 3, :]
        #
        # J[:, 6, :] = x[:, 3, 0, :]
        # J[:, 7, :] = x[:, 3, 1, :]
        # J[:, 14, :] = x[:, 3, 2, :]
        # J[:, 15, :] = x[:, 3, 3, :]

        J11 = torch.zeros((batch, self.na, self.na), device=self.device)
        J22 = torch.zeros((batch, self.na, self.na), device=self.device)
        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -torch.clone(j12)
        J12 = torch.zeros((batch, self.na, self.na), device=self.device)
        J21 = torch.zeros((batch, self.na, self.na), device=self.device)
        J12[:, range(self.na), range(self.na)] = j12
        J21[:, range(self.na), range(self.na)] = j21
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)

        return J - J.transpose(1, 2)


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

        # self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        # self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        # self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        # self.mlp_hidden_4 = MLP(
        #     2 * hidden_dim,
        #     [hidden_dim]
        # ).to(device)

        self.mlp_out = MLP(
            2 * hidden_dim,
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

        # x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1)

        # Q = self.activation_swish(
        #     torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
        #     + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        # K = self.activation_swish(
        #     torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
        #     + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        # V = self.activation_swish(
        #     torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
        #     + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        # x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1).transpose(1, 2)

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

