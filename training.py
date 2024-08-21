from vmas import make_env
from vmas.simulator.utils import save_video
from reinforcement_functions import ReplayBuffer
from functions import LEMURS_qvalue, LEMURS_actor
from functions import PIMARL_qvalue, PIMARL_actor
from functions import MLP_qvalue, MLP_actor
from functions import MSA_qvalue, MSA_actor
from functions import GSA_qvalue, GSA_actor
from parse_args import parse_args
from tqdm import tqdm
from torch.autograd import Variable

from functions import MADDPG
import joblib

import torch
import numpy as np
import PIL


def main(args):
    """ Initial setup """
    cuda_name = "cuda:0"
    preprocessor = True
    folder = "data"
    device = "cpu" if not torch.has_cuda else "cuda"
    wrapper = None
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Create environments """
    env = make_env(
        scenario_name=args.scenario_name,
        num_envs=args.num_envs,
        device=device,
        continuous_actions=True,
        wrapper=wrapper,
        seed=args.seed,
        max_steps=args.max_steps,
        # Environment specific variables
        n_agents=args.n_agents,
        n_agents_good=args.n_agents_good,
        n_agents_adversaries=args.n_agents_adversaries,
        n_packages=1,
        ratio=args.ratio,
    )

    evaluation_env = make_env(
        scenario_name=args.scenario_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        wrapper=wrapper,
        seed=args.seed,
        max_steps=args.max_steps,
        # Environment specific variables
        n_agents=args.n_agents,
        n_agents_good=args.n_agents_good,
        n_agents_adversaries=args.n_agents_adversaries,
        n_packages=1,
        ratio=args.ratio,
    )

    if args.scenario_name == "reverse_transport":
        env.world.landmarks[1].mass = 1
        evaluation_env.world.landmarks[1].mass = 1

    """ Setup actor network """
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        actor_config = {
            "device": device,
            "n_agents": args.n_agents_good + args.n_agents_adversaries,
            "observation_dim_per_agent": env.observation_space[0].shape[0],
            "action_dim_per_agent": env.action_space[0].shape[0],
            "r_communication": args.r_communication,
            "batch_size": args.batch_size,
            "num_envs": args.num_envs,
            "scenario_name": args.scenario_name,
            "preprocessor": preprocessor,
            "ratio": args.ratio,
            "ratio_eval": args.ratio_eval
        }
    else:
        actor_config = {
            "device": device,
            "n_agents": args.n_agents,
            "observation_dim_per_agent": env.observation_space[0].shape[0],
            "action_dim_per_agent": env.action_space[0].shape[0],
            "r_communication": args.r_communication,
            "batch_size": args.batch_size,
            "num_envs": args.num_envs,
            "scenario_name": args.scenario_name,
            "preprocessor": preprocessor,
            "ratio": args.ratio,
            "ratio_eval": args.ratio_eval
        }

    """ Setup Q value networks """
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        qvalue_config = {
            "device": device,
            "n_agents": args.n_agents_good + args.n_agents_adversaries,
            "observation_dim_per_agent": env.observation_space[0].shape[0],
            "action_dim_per_agent": env.action_space[0].shape[0],
            "scenario_name": args.scenario_name
        }
    else:
        qvalue_config = {
            "device": device,
            "n_agents": args.n_agents,
            "observation_dim_per_agent": env.observation_space[0].shape[0],
            "action_dim_per_agent": env.action_space[0].shape[0],
            "scenario_name": args.scenario_name
        }

    if args.neural_network_name == "MLP":
        actor_net = MLP_actor(actor_config)
        q_value_net_1 = MLP_qvalue(qvalue_config)
        q_value_net_2 = MLP_qvalue(qvalue_config)
        q_value_target_net_1 = MLP_qvalue(qvalue_config)
        q_value_target_net_2 = MLP_qvalue(qvalue_config)
    elif args.neural_network_name == "MSA":
        actor_net = MSA_actor(actor_config)
        q_value_net_1 = MSA_qvalue(qvalue_config)
        q_value_net_2 = MSA_qvalue(qvalue_config)
        q_value_target_net_1 = MSA_qvalue(qvalue_config)
        q_value_target_net_2 = MSA_qvalue(qvalue_config)
    elif args.neural_network_name == "GSA":
        actor_net = GSA_actor(actor_config)
        q_value_net_1 = GSA_qvalue(qvalue_config)
        q_value_net_2 = GSA_qvalue(qvalue_config)
        q_value_target_net_1 = GSA_qvalue(qvalue_config)
        q_value_target_net_2 = GSA_qvalue(qvalue_config)
    else:
        if args.scenario_name == "simple_spread" or args.scenario_name == "reverse_transport" or args.scenario_name == "sampling":
            actor_net = LEMURS_actor(actor_config)
            q_value_net_1 = LEMURS_qvalue(qvalue_config)
            q_value_net_2 = LEMURS_qvalue(qvalue_config)
            q_value_target_net_1 = LEMURS_qvalue(qvalue_config)
            q_value_target_net_2 = LEMURS_qvalue(qvalue_config)
        else:
            actor_net = PIMARL_actor(actor_config)
            q_value_net_1 = PIMARL_qvalue(qvalue_config)
            q_value_net_2 = PIMARL_qvalue(qvalue_config)
            q_value_target_net_1 = PIMARL_qvalue(qvalue_config)
            q_value_target_net_2 = PIMARL_qvalue(qvalue_config)

    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        adversarial_actors = []
        for i in range(args.n_agents_adversaries):
            weights_dict = joblib.load("adversaries/"+args.scenario_name[:-5]+"/agent"+str(np.random.randint(3))+".weights")
            adv_net = MADDPG(39, 5, i, device).to(device)
            counter = 0
            for param in zip(adv_net.parameters()):
                if counter % 2 == 0:
                    param[0].data = torch.from_numpy(weights_dict['p_variables'][counter]).transpose(0, 1).to(device)
                else:
                    param[0].data = torch.from_numpy(weights_dict['p_variables'][counter]).to(device)
                counter += 1
            adversarial_actors.append(adv_net.eval())

    q_value_net_1_weights = q_value_net_1.state_dict()
    q_value_net_2_weights = q_value_net_2.state_dict()

    q_value_target_net_1.load_state_dict(q_value_net_1_weights)
    q_value_target_net_2.load_state_dict(q_value_net_2_weights)

    """ Setup Collector """
    collector_config = {
        # Lower bound of the total number of frames returned by the collector.
        # The iterator will stop once the total number of frames equates or exceeds the
        # total number of frames passed to the collector.
        "total_frames": args.total_frames,
        # Time-length of a batch. reset_at_each_iter and frames_per_batch == n_steps_max are
        # equivalent configurations. default: 200
        "frames_per_batch": args.frames_per_batch,
        # Number of frames for which the policy is ignored before it is called.
        # This feature is mainly intended to be used in offline/model-based settings,
        # where a batch of random trajectories can be used to initialize training.
        # default=-1 (i.e. no random frames)
        "init_random_frames": args.init_random_frames,
        # The device on which the policy will be placed. If it differs from the input
        # policy device, the update_policy_weights_() method should be queried at
        # appropriate times during the training loop to accommodate for the lag between
        # parameter configuration at various times. default = None
        # (i.e. policy is kept on its original device)
        "device": device,
        # Seed to be used for torch and numpy.
        "seed": args.seed,
    }

    """ Setup Loss Module """
    sac_config = {  # Discount for return computation Default is 0.99
        "gamma": args.gamma,
        # Initial entropy multiplier. Default is 1.0.
        "alpha_init": args.alpha_init,
        "alpha_min": torch.Tensor([args.min_alpha]).to(device),
        "alpha_max": torch.Tensor([args.max_alpha]).to(device),
        "alpha": torch.Tensor([args.alpha_init]).to(device),
        "lr_alpha": args.lr_alpha,
        "target_entropy": -env.action_space[0].shape[0],
        # Tau
        "tau": args.tau,
        # Reward scaling
        "reward_scaling": args.reward_scaling
    }
    sac_config["alpha"].requires_grad = True

    """ Setup Optimizer """
    optimizer_policy = torch.optim.Adam(actor_net.parameters(), args.lr)
    optimizer_q_value = torch.optim.Adam(list(q_value_net_1.parameters()) + list(q_value_net_2.parameters()), args.lr)
    optimizer_alpha = torch.optim.Adam([sac_config["alpha"]], sac_config["lr_alpha"])

    """ Setup Replay Buffer """
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        replay_buffer = ReplayBuffer(
            env.observation_space[0].shape[0] * (args.n_agents_good + args.n_agents_adversaries),
            env.action_space[0].shape[0] * (args.n_agents_good + args.n_agents_adversaries),
            args.num_envs,
            args.n_agents_good + args.n_agents_adversaries,
            args.max_size,
            device
        )
    else:
        replay_buffer = ReplayBuffer(
            env.observation_space[0].shape[0] * args.n_agents,
            env.action_space[0].shape[0] * args.n_agents,
            args.num_envs,
            args.n_agents,
            args.max_size,
            device
        )

    """ Setup explorer """
    exploration_process_noise_config = {
        "num_envs": args.num_envs,
        "num_agents": args.n_agents,
        "action_dim_per_agent": env.action_space[0].shape[0]
    }

    """ Setup Trainer """
    trainer_config = {
        # Total number of frames to be collected during training. ~1M is reasonable
        "total_frames": args.total_frames,
        #  Number of optimization steps per collection of data. A trainer works as follows: a main loop
        #  collects batches of data (epoch loop), and a sub-loop (training loop) performs model updates
        #  in between two collections of data. Default is 500
        "optim_steps_per_batch": args.optim_steps_per_batch,
        # If True, the gradients will be clipped based on the total norm of the model parameters.
        # If False, all the partial derivatives will be clamped to (-clip_norm, clip_norm).
        # Default is True.
        "clip_grad_norm": args.clip_grad_norm,
        # Value to be used for clipping gradients. Default is 100.0.
        "clip_norm": args.clip_norm,
        # If True, a progress bar will be displayed using tqdm. If tqdm is not installed,
        # this option won’t have any effect. Default is True
        "progress_bar": True,
        # Seed to be used for the collector, pytorch and numpy. Default is 42.
        "seed": args.seed,
        # How often the trainer should be saved to disk. Default is 10000.
        "save_trainer_interval": args.save_trainer_interval,
    }

    """ Training loop """
    actor_net.eval()
    q_value_net_1.eval()
    q_value_net_2.eval()
    obs = [None for i in range(args.num_envs)]
    dones = [False]
    MEAN_CUM_REWS = []
    STD_CUM_REWS = []

    for current_frame in tqdm(range(trainer_config["total_frames"])):

        # Interaction with the environment
        with torch.no_grad():

            # Preprocess obs to a list of tensors for a proper reset
            obs = list(obs)

            # Reset environment if needed
            if current_frame == 0:
                for environment in range(args.num_envs):
                    obs[environment] = torch.stack(env.reset_at(environment)).squeeze(1)
            if any(dones):
                environments = [i for i, x in enumerate(dones) if x]
                for environment in environments:
                    obs[environment] = torch.stack(env.reset_at(environment)).squeeze(1)

            # Post process observation list to obtain a Tensor
            obs = torch.stack(obs)

            # If not enough data in the replay buffer
            if current_frame < collector_config["init_random_frames"]:

                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    # Random actions
                    adversarial_actions = torch.stack(
                        [
                            adversarial_actors[i](obs) for i in range(args.n_agents_adversaries)
                        ], dim=1
                    )
                    actions = torch.cat((adversarial_actions.to("cpu"),
                                         (torch.rand(exploration_process_noise_config["num_envs"],
                                                     exploration_process_noise_config["num_agents"],
                                                     exploration_process_noise_config["action_dim_per_agent"]
                                                     ) - 0.5
                                         ) * 2.0),
                                        dim=1)
                else:
                    actions = (torch.rand(exploration_process_noise_config["num_envs"],
                                          exploration_process_noise_config["num_agents"],
                                          exploration_process_noise_config["action_dim_per_agent"]
                                          ) - 0.5
                               ) * 2.0

            else:
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    # Select action
                    adversarial_actions = torch.stack(
                        [
                            adversarial_actors[i](obs) for i in range(args.n_agents_adversaries)
                        ], dim=1
                    )
                    good_actions = actor_net(obs).rsample()
                    actions = torch.cat((adversarial_actions, good_actions), dim=1)
                else:
                    actions = actor_net(obs).rsample()

            # Execute action in the environment, receiving the next state, the reward, and the done signals
            next_obs, rews, dones, info = env.step(list(actions.transpose(0, 1)))

            # Update replay buffer with the new experiences
            replay_buffer.add(obs.detach().cpu().numpy(),
                              actions.detach().cpu().numpy(),
                              torch.stack(rews).transpose(0, 1).detach().cpu().numpy(),
                              torch.stack(next_obs).transpose(0, 1).detach().cpu().numpy(),
                              dones.detach().cpu().numpy())

            # Update obs
            obs = torch.stack(next_obs).transpose(0, 1).clone()

        # If it is time to update
        if current_frame % args.frames_per_batch == 0 and current_frame >= collector_config["init_random_frames"]:

            # Networks now train
            actor_net.train()
            q_value_net_1.train()
            q_value_net_2.train()

            # Store rewards for checking purposes
            rb_rewards = []
            for iteration in range(trainer_config["optim_steps_per_batch"]):

                # Randomly sample a batch of transitions according to the prioritized replay buffer
                b_obs, b_actions, b_rews, b_next_obs, b_dones = replay_buffer.sample(args.batch_size)

                # Compute targets for the Q functions
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    adversarial_actions = torch.stack(
                        [
                            adversarial_actors[i](b_next_obs) for i in range(args.n_agents_adversaries)
                        ], dim=1
                    )
                    good_action_distribution = actor_net(b_next_obs)
                    good_actions = good_action_distribution.rsample()
                    actions = torch.cat((adversarial_actions, good_actions), dim=1)
                    log_probs = good_action_distribution.log_prob(good_actions ).sum(dim=-1).sum(dim=-1)
                else:
                    action_distribution = actor_net(b_next_obs)
                    actions = action_distribution.rsample()
                    log_probs = action_distribution.log_prob(actions ).sum(dim=-1).sum(dim=-1)
                q_value_target_1 = q_value_target_net_1(b_next_obs, actions).flatten()
                q_value_target_2 = q_value_target_net_2(b_next_obs, actions).flatten()

                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    b_target = sac_config["reward_scaling"] * b_rews[:, args.n_agents_adversaries:].sum(dim=1) \
                               + sac_config["gamma"] * (1 - b_dones.flatten()) * \
                               (torch.minimum(q_value_target_1, q_value_target_2) - sac_config["alpha"] * log_probs)
                else:
                    b_target = sac_config["reward_scaling"] * b_rews.sum(dim=1) \
                               + sac_config["gamma"] * (1 - b_dones.flatten()) * \
                               (torch.minimum(q_value_target_1, q_value_target_2) - sac_config["alpha"] * log_probs)

                # Update Q-functions by one step gradient descent
                q_value_1 = q_value_net_1(b_obs, b_actions).flatten()
                q_value_2 = q_value_net_2(b_obs, b_actions).flatten()
                loss_q_value_1 = (q_value_1 - b_target).pow(2).mean()
                loss_q_value_2 = (q_value_2 - b_target).pow(2).mean()
                loss_q_value = loss_q_value_1 + loss_q_value_2

                optimizer_q_value.zero_grad()

                loss_q_value.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(q_value_net_1.parameters(), trainer_config["clip_norm"])
                    torch.nn.utils.clip_grad_norm_(q_value_net_2.parameters(), trainer_config["clip_norm"])

                optimizer_q_value.step()

                # Update policy by one step gradient descent
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    adversarial_actions = torch.stack(
                        [
                            adversarial_actors[i](b_obs) for i in range(args.n_agents_adversaries)
                        ], dim=1
                    )
                    good_action_distribution = actor_net(b_obs)
                    good_actions = good_action_distribution.rsample()
                    actions = torch.cat((adversarial_actions, good_actions), dim=1)
                    log_probs = good_action_distribution.log_prob(good_actions ).sum(dim=-1).sum(dim=-1)
                else:
                    action_distribution = actor_net(b_obs)
                    actions = action_distribution.rsample()
                    log_probs = action_distribution.log_prob(actions ).sum(dim=-1).sum(dim=-1)
                q_value_1 = q_value_net_1(b_obs, actions).flatten()
                q_value_2 = q_value_net_2(b_obs, actions).flatten()
                loss_policy = (sac_config["alpha"] * log_probs - torch.minimum(q_value_1, q_value_2)).mean()

                optimizer_policy.zero_grad()

                loss_policy.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), trainer_config["clip_norm"])


                optimizer_policy.step()

                # Update SAC temperature
                log_probs_alpha = Variable(log_probs.data, requires_grad=True)

                alpha_loss = (sac_config["alpha"] * (-log_probs_alpha - sac_config["target_entropy"])).mean()

                optimizer_alpha.zero_grad()

                alpha_loss.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(sac_config["alpha"], trainer_config["clip_norm"])

                optimizer_alpha.step()

                if sac_config["alpha"] < sac_config["alpha_min"]:
                    sac_config["alpha"] = sac_config["alpha_min"]
                    sac_config["alpha"].requires_grad = True

                if sac_config["alpha"] > sac_config["alpha_max"]:
                    sac_config["alpha"] = sac_config["alpha_max"]
                    sac_config["alpha"].requires_grad = True

                # Update target networks
                for param, target_param in zip(q_value_net_1.parameters(), q_value_target_net_1.parameters()):
                    target_param.data.copy_(sac_config["tau"] * param.data + (1 - sac_config["tau"]) * target_param.data)

                for param, target_param in zip(q_value_net_2.parameters(), q_value_target_net_2.parameters()):
                    target_param.data.copy_(sac_config["tau"] * param.data + (1 - sac_config["tau"]) * target_param.data)

                # Store rewards
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    rb_rewards.append(sac_config["reward_scaling"] * b_rews[:, args.n_agents_adversaries:].sum(dim=1).mean())
                else:
                    rb_rewards.append(sac_config["reward_scaling"] * b_rews.sum(dim=1).mean())

        # Networks stop to train
        actor_net.eval()
        q_value_net_1.eval()
        q_value_net_2.eval()

        # Save target weights
        if current_frame % args.save_trainer_interval == 0 and current_frame >= collector_config["init_random_frames"]:
            torch.save(q_value_target_net_1, folder+'/q_value_target_net_1_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.pth')
            torch.save(q_value_target_net_2, folder+'/q_value_target_net_2_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.pth')
            torch.save(q_value_net_1, folder+'/q_value_net_1_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.pth')
            torch.save(q_value_net_2, folder+'/q_value_net_2_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.pth')
            torch.save(actor_net, folder+'/actor_net_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.pth')

            # Display some metrics
            print("\n-------------------------------------------\n")
            print(f"Training instance {int(current_frame / args.frames_per_batch)}")
            print("")
            print(f"Policy Loss after the optimization steps: {loss_policy.item()}")
            print(f"Q1 Loss after the optimization steps: {loss_q_value_1.item()}")
            print(f"Q2 Loss after the optimization steps: {loss_q_value_2.item()}")
            print(f"Reward: {(sac_config['reward_scaling'] * rb_rewards[-1]).item()}")
            print(f"Alpha: {sac_config['alpha'].item()}")
            print("\n-------------------------------------------\n")

        # Evaluate if considered
        eval_steps = args.max_frames_eval

        if current_frame % args.evaluation_interval == 0 and current_frame >= collector_config["init_random_frames"]:
            cum_rew = np.zeros([args.num_test_episodes])
            with torch.no_grad():
                for episode in range(args.num_test_episodes):
                    time_step = 0
                    eval_dones = [False]
                    eval_obs = evaluation_env.reset(seed=None)
                    eval_cum_rew = 0
                    while not any(eval_dones) and time_step < eval_steps:
                        eval_obs = torch.stack(eval_obs)
                        if len(eval_obs.shape) == 3:
                            eval_obs = eval_obs.squeeze(1)
                        eval_obs = eval_obs.unsqueeze(0)
                        if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                            # Select action
                            eval_adversarial_actions = torch.stack(
                                [
                                    adversarial_actors[i](eval_obs) for i in range(args.n_agents_adversaries)
                                ], dim=1
                            )
                        eval_action_distribution = actor_net(eval_obs)
                        eval_actions = eval_action_distribution.mean
                        if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                            eval_actions = torch.cat((eval_adversarial_actions, eval_actions), dim=1)
                        eval_obs, eval_rews, eval_dones, eval_info = evaluation_env.step(list(eval_actions.transpose(0, 1)))
                        time_step += 1
                        if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                            eval_cum_rew += (sac_config["reward_scaling"] * torch.stack(eval_rews[args.n_agents_adversaries:]).sum()).item()
                        else:
                            eval_cum_rew += (sac_config["reward_scaling"] * torch.stack(eval_rews).sum()).item()
                        cum_rew[episode] = eval_cum_rew

            mean_cum_rew = np.mean(cum_rew)
            std_cum_rew = np.std(cum_rew)
            print("\n-------------------------------------------\n")
            print(f"Evaluation instance {int(current_frame / args.evaluation_interval)}")
            print("")
            print(f"Cumulative Reward: mean is {mean_cum_rew} and std is {std_cum_rew}")
            print("\n-------------------------------------------\n")
            MEAN_CUM_REWS.append(mean_cum_rew)
            STD_CUM_REWS.append(std_cum_rew)
            np.save(folder+'/MEAN_CUM_REWS_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.npy', np.array(MEAN_CUM_REWS))
            np.save(folder+'/STD_CUM_REWS_'+args.scenario_name+'_'+str(current_frame)+'_'+str(args.neural_network_name)+'.npy', np.array(STD_CUM_REWS))


if __name__ == "__main__":
    main(parse_args())
