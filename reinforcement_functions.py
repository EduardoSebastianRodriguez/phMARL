"""
This file contains basic reinforcement learning tools. Should not be touched.

(1) TanhTransform and SquashedNormal modify the actor policy output to match a diagonal gaussian.

(2) ReplayBuffer is a class that implements a uniformly sampled replay buffer.
"""

import random
import numpy as np
import torch

from torch import distributions as pyd
import math
import torch.nn.functional as F


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, num_envs, num_agents, capacity, device):
        self.capacity = capacity
        self.device = device
        self.num_envs = num_envs
        self.num_agents = num_agents

        self.obses = np.empty((capacity, obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, num_agents), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        for _ in range(self.num_envs):
            np.copyto(self.obses[self.idx], obs[_].flatten())
            np.copyto(self.actions[self.idx], action[_].flatten())
            np.copyto(self.rewards[self.idx], reward[_].flatten())
            np.copyto(self.next_obses[self.idx], next_obs[_].flatten())
            np.copyto(self.dones[self.idx], done[_])

            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float().reshape(batch_size, self.num_agents, -1)
        actions = torch.as_tensor(self.actions[idxs], device=self.device).reshape(batch_size, self.num_agents, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float().reshape(batch_size, self.num_agents, -1)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones


