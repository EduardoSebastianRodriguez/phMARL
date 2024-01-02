import numpy as np
import torch
from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
import os


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.num_good = kwargs.get("n_agents_good", 6)
        self.num_adversaries = kwargs.get("n_agents_adversaries", 6)
        self.num_agents = self.num_adversaries + self.num_good
        self.obs_agents = kwargs.get("obs_agents", True)
        self.live = torch.ones(
            batch_dim, self.num_agents, device=device, dtype=torch.float32
        )
        self.ratio = kwargs.get("ratio", 5)  # ratio = 3, 4, 5
        self.device = device

        self.num_landmarks = self.num_good

        world = World(batch_dim=batch_dim, device=device)

        # Add agents
        for i in range(self.num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=(0.075 if i < self.num_adversaries else 0.05)),
                color=(Color.RED if i < self.num_adversaries else Color.BLUE),
                adversary=(True if i < self.num_adversaries else False),
                max_speed=(2.0 if i < self.num_adversaries else 3.0),
                movable=True,
                u_range=4.0
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(self.num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=0.03),
                color=Color.GREEN,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                self.ratio
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 0.5 * self.ratio,
                batch_index=env_index,
            )
            agent.color = Color.RED if agent.adversary else Color.BLUE
            agent.mass = 1.0
            if env_index == None:
                self.live[:, int(agent.name[6:])] = 1
            else:
                self.live[env_index, int(agent.name[6:])] = 1

        for landmark in self.world.landmarks:
            landmark.set_pos(
                self.ratio
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 0.5 * self.ratio,
                batch_index=env_index,
            )

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )

    def agent_reward(self, agent: Agent):
        pos_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        closest = torch.min(
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        landmark.state.pos - agent.state.pos, dim=1
                    )
                    for landmark in self.world.landmarks
                ],
                dim=1,
            ),
            dim=-1,
        )[0]
        pos_rew[self.live[:, int(agent.name[6:])] == 1.0] -= closest[self.live[:, int(agent.name[6:])] == 1.0]

        for landmark in self.world.landmarks:
            found = self.world.is_overlapping(landmark, agent) & (self.live[:, int(agent.name[6:])] == 1.0)
            pos_rew[found] += 20
            while torch.where(found)[0].shape[0] != 0:
                landmark.set_pos(
                    self.ratio
                    * torch.rand(
                        self.world.dim_p, device=self.world.device, dtype=torch.float32
                    )
                    - 0.5 * self.ratio,
                    batch_index=torch.where(found)[0][0],
                )
                found = self.world.is_overlapping(landmark, agent) & (self.live[:, int(agent.name[6:])] == 1.0)

        if agent.collide:
            for a in self.world.agents:
                if a != agent and a.adversary:
                    killed = self.world.is_overlapping(a, agent) & (self.live[:, int(agent.name[6:])] == 1.0)
                    pos_rew[killed] -= 5
                    self.live[killed, int(agent.name[6:])] = 0
                    if killed.shape[0] == 1 and killed[0] == True: 
                        agent.color = Color.GRAY
                        agent.mass = 1000000            
        return pos_rew

    def adversary_reward(self, agent: Agent):
        adv_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        closest = torch.min(
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        agent.state.pos - evader.state.pos, dim=1
                    )
                    for evader in self.good_agents()
                ],
                dim=1,
            ),
            dim=-1,
        )[0]
        adv_rew -= closest

        if agent.collide:
            for a in self.world.agents:
                if a != agent and not a.adversary:
                    killed = self.world.is_overlapping(a, agent) & (self.live[:, int(agent.name[6:])] == 1.0)
                    adv_rew[killed] += 15

        return adv_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        distance = 10000
        landmark_pos = []
        if agent.adversary:
            landmark_pos.append(self.world.landmarks[int(agent.name[6:])].state.pos - agent.state.pos)
        else:
            for landmark in self.world.landmarks:  # world.entities:
                if distance > torch.linalg.norm(landmark.state.pos - agent.state.pos):
                    distance = torch.linalg.norm(landmark.state.pos - agent.state.pos)
                    vector = landmark.state.pos - agent.state.pos
            landmark_pos.append(vector)

        type_adv = []
        for index in range(agent.state.pos.shape[0]):
            type_adv.append(float(agent.adversary))
        return torch.cat(
            [agent.state.pos, agent.state.vel, self.live[:, int(agent.name[6:])].unsqueeze(1), torch.Tensor(type_adv).to(self.device).unsqueeze(1), *landmark_pos],
            dim=-1
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
