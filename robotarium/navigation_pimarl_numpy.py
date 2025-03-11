#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import matplotlib.patches as patches
import matplotlib
import matplotlib.animation as animation

import torch
import numpy as np
from functions import LEMURS_actor, PIMARL_actor, laplacian

# Initial setup
device = "cpu" 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Change parameters here for evaluation
num_agents_eval = 4
observation_dim_per_agent = 6 # 2 for position, 2 for velocity, 2 for desired position

# Setup actor network
# Load model, set "scenario_name" to "navigation" or something different to "simple_spread"
# By default, dt = 0.1 in VMAS (sometimes set to 0.05); in Robotarium, dt=0.033
actor_config = {
            "device": device,
            "n_agents": num_agents_eval,
            "observation_dim_per_agent": observation_dim_per_agent,
            "action_dim_per_agent": 2,
            "r_communication": 0.75,
            "batch_size": 1,
            "num_envs": 1,
            "preprocessor": True,
            "ratio": 2,
            "ratio_eval": 2,
            "scenario_name": "navigation_robotarium",
        }

actor_net_weights = np.load('actor_net_improved.npy', allow_pickle=True)
actor_net_weights_pytorch = {k: torch.from_numpy(v) for k, v in actor_net_weights.item().items()}
actor_net = PIMARL_actor(actor_config) 
actor_net.load_state_dict(actor_net_weights_pytorch)
actor_net.eval()
r_communication = actor_net.r_communication

# Experiment constants
iterations = 900
magnitude_limit = 0.15
dt = 0.033
goal_reached_threshold = 0.075

# Robotarium setup
r = robotarium.Robotarium(number_of_robots=num_agents_eval, show_figure=True, sim_in_real_time=False)
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(safety_radius=0.13)
si_to_uni_dyn = create_si_to_uni_dynamics()

# Initializations
delay_rate = 0.0 
packet_rate = 0.0
disturbance_rate = 0.0
disturbance_std = 0.00
T_delays = 10

# Logging
travel_times = np.zeros([num_agents_eval])
travel_distances = np.zeros([num_agents_eval])
success = np.zeros([num_agents_eval])

# Colormap
cmap = matplotlib.colormaps.get_cmap('nipy_spectral')
traces = []
colors = [cmap(i / num_agents_eval) for i in range(num_agents_eval)]

# Change color of the chassis of the robots
# for i in range(num_agents_eval):
#     r.chassis_patches[i].set_facecolor(colors[i])

with torch.no_grad():
    lines = []
    delay_free = [[0 for j in range(num_agents_eval)] for i in range(num_agents_eval)]

    X = np.zeros((2, num_agents_eval))
    V = np.zeros((2, num_agents_eval))
    X_desired = (np.random.rand(2, num_agents_eval) - np.array([[0.5], [0.5]])) * np.array([[2.8], [1.4]])
    X_prev = np.copy(X)
    desired_correct = False
    while not desired_correct:
        desired_correct = True
        for i in range(num_agents_eval):
            for j in range(num_agents_eval):
                if i != j and np.linalg.norm(X_desired[:, i] - X_desired[:, j]) < 0.25:
                    X_desired[:, j] = (np.random.rand(2) - np.array([0.5, 0.5])) * np.array([2.8, 1.4])
                    desired_correct = False
                    break
                else:
                    continue

    obs = np.concatenate((X, V, X_desired - X), axis=0)
    obs_full = np.expand_dims(obs, axis=0).repeat(num_agents_eval, axis=0)
    obs_delays = np.zeros((T_delays, observation_dim_per_agent, num_agents_eval))   
 
    for k in range(iterations):

        # Get the poses of the robots
        x = r.get_poses()

        # Update the observation matrix
        X_prev = np.copy(X)
        X = x[:2, :]
        obs = np.concatenate((X, V, X_desired - X), axis=0)
        obs_full = np.expand_dims(obs, axis=0).repeat(num_agents_eval, axis=0)
        traces.append(np.copy(x[:2, :]))

        L = laplacian(torch.from_numpy(np.expand_dims(X, axis=0)).transpose(1, 2), r_communication, num_agents_eval, device).squeeze(0)
        current_L = L.numpy()

        # Update graphics
        while len(lines) > 0:
            lines.pop(0).remove()

        # Modify communication topology depending on disturbances, packet losses and delays
        counter = 0
        for i in range(num_agents_eval):
            neighbors_indeces = np.where(L.numpy()[i, :] != 0)[0] 
            if len(neighbors_indeces) > 0:
                for neighbor_index in neighbors_indeces:
                    flag_delays = np.random.binomial(1, delay_rate)
                    flag_packet = np.random.binomial(1, packet_rate)
                    flag_disturbances = np.random.binomial(1, disturbance_rate)

                    # 1. Delays
                    if delay_free[i][neighbor_index] > 0:
                        delay_free[i][neighbor_index] -= 1
                    if flag_delays and delay_free[i][neighbor_index] == 0: 
                        delay = np.random.choice(np.linspace(0, T_delays-1, T_delays, dtype=int))
                        obs_full[i, :, neighbor_index] = obs_delays[delay, :, neighbor_index]
                        delay_free[i][neighbor_index] = delay

                    # 2. Packet Losses
                    if flag_packet:
                        current_L[i, neighbor_index] = 0

                    # 3. Disturbances
                    if flag_disturbances:
                        obs_full[i, :, neighbor_index] += np.random.multivariate_normal(np.zeros(observation_dim_per_agent), disturbance_std * np.eye(observation_dim_per_agent))

                    if flag_delays or delay_free[i][neighbor_index] > 0:
                        lines.append(plt.plot([x[0, i], x[0, neighbor_index]], [x[1, i], x[1, neighbor_index]], 'm-', zorder=0)[0])
                    elif flag_packet:
                        lines.append(plt.plot([x[0, i], x[0, neighbor_index]], [x[1, i], x[1, neighbor_index]], 'r-', zorder=0)[0])
                    elif flag_disturbances:
                        lines.append(plt.plot([x[0, i], x[0, neighbor_index]], [x[1, i], x[1, neighbor_index]], 'y-', zorder=0)[0])
                    else:
                        lines.append(plt.plot([x[0, i], x[0, neighbor_index]], [x[1, i], x[1, neighbor_index]], 'g-', zorder=0)[0])
        # Plot desired positions
        trace = np.array(traces)
        for i in range(num_agents_eval):
            lines.append(plt.plot(X_desired[0, i], X_desired[1, i], 'o', markersize=40, zorder=0, color=colors[i])[0])
            lines.append(plt.plot(X_desired[0, i], X_desired[1, i], 'x', markersize=36, zorder=0, color='w')[0])
            lines.append(plt.plot(trace[:, 0, i], trace[:, 1, i], linewidth=3, color=colors[i], zorder=0)[0])
        
        plt.gcf().canvas.flush_events()

        eval_action_distribution = actor_net(torch.from_numpy(obs_full).transpose(1, 2), torch.from_numpy(current_L))
        eval_actions = eval_action_distribution.mean[0, :, :].transpose(0, 1).cpu().numpy()

        # Update velocity vector
        V += eval_actions*dt

        #Keep single integrator control vectors under specified magnitude
        # Threshold control inputs
        norms = np.linalg.norm(V, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        V[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        # Make sure that the robots don't collide
        V = si_barrier_cert(V, x[:2, :])

        # If robots are close to their destination, stop them
        for i in range(num_agents_eval):
            if np.linalg.norm(X_desired[:, i] - X[:, i]) < goal_reached_threshold:
                V[:, i] = np.zeros(2)
                success[i] = 1

        # Logg info
        for i in range(num_agents_eval):
            travel_distances[i] += np.linalg.norm(X[:, i] - X_prev[:, i])
            if success[i] == 0:
                travel_times[i] += dt

        # Transform the single-integrator dynamcis to unicycle dynamics
        dxu = si_to_uni_dyn(V, x)

        # Set the velocities of the robots
        r.set_velocities(np.arange(num_agents_eval), dxu)

        # Iterate the simulation
        r.step()

        # Update delay matrix
        obs_delays[1:T_delays, :, :] = obs_delays[0:T_delays-1, :, :]
        obs_delays[0, :, :] = obs 

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

# Report numbers
print("Average travel distances: ", np.mean(travel_distances))
print("Average travel times: ", np.mean(travel_times))
print("Success rate: ", np.mean(success))
print("Variance travel distances: ", np.std(travel_distances))
print("Variance travel times: ", np.std(travel_times))
print("Variance success rate: ", np.std(success))

# Save the average travel distance, travel time and success rate
np.save('average_travel_distances.npy', np.mean(travel_distances))
np.save('average_travel_times.npy', np.mean(travel_times))
np.save('average_success.npy', np.mean(success))
np.save('variance_travel_distances.npy', np.std(travel_distances))
np.save('variance_travel_times.npy', np.std(travel_times))
np.save('variance_success.npy', np.std(success))
