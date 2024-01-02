from vmas import make_env
from parse_args import parse_args

import torch
import numpy as np
import PIL.Image

import joblib


def main(args):
    """ Initial setup """
    cuda_name = "cuda:0"
    folder = "data"
    preprocessor = True
    device = "cpu" if not torch.has_cuda else cuda_name
    wrapper = None
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Change parameters here for evaluation """
    evaluation_env = make_env(
        scenario_name=args.scenario_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        wrapper=wrapper,
        seed=args.seed,
        max_steps=args.max_frames_eval,
        # Environment specific variables
        n_agents=args.num_agents_eval,
        n_agents_good=args.n_agents_good_eval,
        n_agents_adversaries=args.n_agents_adversaries_eval,
        n_packages=1,
        share_reward=True,
        ratio=args.ratio_eval,
    )
    if args.scenario_name == "reverse_transport":
        evaluation_env.world.landmarks[1].mass = 1

    """ Setup actor network """
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        actor_config = {
            "device": device,
            "n_agents": args.n_agents_good_eval + args.n_agents_adversaries_eval,
            "observation_dim_per_agent": evaluation_env.observation_space[0].shape[0],
            "action_dim_per_agent": evaluation_env.action_space[0].shape[0],
            "r_communication": args.r_communication,
            "batch_size": args.batch_size,
            "num_envs": args.num_envs,
            "preprocessor": preprocessor,
            "ratio": args.ratio,
            "ratio_eval": args.ratio_eval
        }
    else:
        actor_config = {
            "device": device,
            "n_agents": args.num_agents_eval,
            "observation_dim_per_agent": evaluation_env.observation_space[0].shape[0],
            "action_dim_per_agent": evaluation_env.action_space[0].shape[0],
            "r_communication": args.r_communication,
            "batch_size": args.batch_size,
            "num_envs": args.num_envs,
            "preprocessor": preprocessor,
            "ratio": args.ratio,
            "ratio_eval": args.ratio_eval
        }
    sac_config = {
        # Reward scaling
        "reward_scaling": args.reward_scaling
    }

    if args.neural_network_name == "MLP":
        from functions import MLP_actor
    elif args.neural_network_name == "MSA":
        from functions import MSA_actor
    elif args.neural_network_name == "GSA":
        from functions import GSA_actor
    else:
        args.neural_network_name = "LEMURS"
        if args.scenario_name == "simple_spread" or args.scenario_name == "reverse_transport" or args.scenario_name == "sampling":
            from functions import LEMURS_actor
        else:
            from functions import PIMARL_actor

    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        from functions import MADDPG

    actor_net = torch.load(folder+'/actor_net_'+args.scenario_name+'_'+str(args.desired_frame)+'_'+str(args.neural_network_name)+'.pth')
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        actor_net.na = args.n_agents_good_eval + args.n_agents_adversaries_eval
    else:
        actor_net.na = args.num_agents_eval
    actor_net.eval()
    if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
        adversarial_actors = []
        for i in range(args.n_agents_adversaries_eval):
            weights_dict = joblib.load(
                "adversaries/" + args.scenario_name[:-5] + "/agent" + str(np.random.randint(3)) + ".weights")
            adv_net = MADDPG(39, 5, i, device).to(device)
            counter = 0
            for param in zip(adv_net.parameters()):
                if counter % 2 == 0:
                    param[0].data = torch.from_numpy(weights_dict['p_variables'][counter]).transpose(0, 1).to(device)
                else:
                    param[0].data = torch.from_numpy(weights_dict['p_variables'][counter]).to(device)
                counter += 1
            adversarial_actors.append(adv_net.eval())
        trajectories = np.zeros([args.max_frames_eval, args.n_agents_good_eval + args.n_agents_adversaries_eval, 2])
    else:
        trajectories = np.zeros([args.max_frames_eval, args.num_agents_eval, 2])
    with torch.no_grad():
        cum_rew = np.zeros([args.num_test_episodes])
        for episode in range(args.num_test_episodes):
            time_step = 0
            frame_list = []
            eval_dones = [False]
            eval_obs = evaluation_env.reset(seed=None)
            eval_cum_rew = 0
            while not any(eval_dones) and time_step < args.max_frames_eval:
                eval_obs = torch.stack(eval_obs)
                if len(eval_obs.shape) == 3:
                    eval_obs = eval_obs.squeeze(1)
                eval_obs = eval_obs.unsqueeze(0)
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    # Select action
                    eval_adversarial_actions = torch.stack(
                        [
                            adversarial_actors[i](eval_obs) for i in range(args.n_agents_adversaries_eval)
                        ], dim=1
                    )
                eval_action_distribution = actor_net(eval_obs)
                eval_actions = eval_action_distribution.mean
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    eval_actions = torch.cat((eval_adversarial_actions, eval_actions), dim=1)
                eval_obs, eval_rews, eval_dones, eval_info = evaluation_env.step(list(eval_actions.transpose(0, 1)))
                time_step += 1
                if args.scenario_name == "grassland_vmas" or args.scenario_name == "adversarial_vmas":
                    eval_cum_rew += (sac_config["reward_scaling"] * torch.stack(
                        #eval_rews).sum()).item()
                        eval_rews[args.n_agents_adversaries_eval:]).sum()).item()
                else:
                    eval_cum_rew += (sac_config["reward_scaling"] * torch.stack(eval_rews).sum()).item()
                cum_rew[episode] = eval_cum_rew
                trajectories[time_step, :, :] = torch.stack(eval_obs)[:, 0, 0:2].detach().cpu().numpy()
                time_step += 1
                frame_list.append(PIL.Image.fromarray(evaluation_env.render(mode="rgb_array",
                                                                            visualize_when_rgb=True)))

            ###########################################################################
            # UNCOMMENT THIS PART OF THE CODE IF YOU WANT TO SAVE THE ANIMATIONS
            """
            frame_list[0].save('animations/' +
                               args.scenario_name +
                               '_' +
                               str(args.num_agents_eval) +
                               '_' +
                               str(episode) +
                               '_' +
                               str(args.neural_network_name) +
                               '.gif',
                               save_all=True, append_images=frame_list[1:], duration=3, loop=0)
            np.save('data/trajectories' +
                    args.scenario_name +
                    '_' +
                    str(args.num_agents_eval) +
                    '_' +
                    str(episode) +
                    '_' +
                    str(args.neural_network_name) +
                    '.npy', trajectories)
            """
            ###########################################################################

        mean_cum_rew = np.mean(cum_rew)
        std_cum_rew = np.std(cum_rew)
        print("\n-------------------------------------------\n")
        print(f"Evaluation instance {episode}")
        print("")
        print(f"Cumulative Reward: mean is {mean_cum_rew} and std is {std_cum_rew}")
        print("\n-------------------------------------------\n")


if __name__ == "__main__":
    main(parse_args())
