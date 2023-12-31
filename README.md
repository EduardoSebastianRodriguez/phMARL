# Physics-Informed Multi-Agent Reinforcement Learning for Distributed Multi-Robot Problems

This repository contains all the supplementary material 
associated to the paper with title "Physics-Informed Multi-Agent 
Reinforcement Learning for Distributed Multi-Robot Problems". 
Please check out our project website for more details: https://eduardosebastianrodriguez.github.io/phMARL/.

## Dependencies

Our code is tested with Ubuntu 20.04 and Python 3.10. It depends on the following Python packages: 

```torch 1.12.1``` 

```numpy 1.24.3```

```matplotlib 3.7.1```

## Some qualitative results

We propose a novel MARL approach to learn distributed-by-design control policies for general cooperative/competitive multi-robot tasks. 
The solution has three key characteristics: (1) we use a port-Hamiltonian description of the multi-robot system and task that respects the network
topology and robot energy conservation laws, leading to a scalable and sampling efficient parameterization
of the control policy; (2) we parameterize the control policy using self-attention neural networks that handle the time-varying
information available at each robot, so we are able to learn the task while achieving invariance with respect to the number of
robots in the multi-robot team; and (3) we propose modifications over a soft actor-critic to overcome non-stationarity issues while 
allowing robots to interact with other members of the team, and also avoid value factorization/approximations.

In the following we show some qualitative results from six cooperative/competitive scenarios. They cover a wide variety of cooperative and competitive behaviors such as collision avoidance, navigation, transport, evasion and monitoring.
All the policies are trained with 4 robots.

### Reverse transport

|                     4 robots                      |                    8 robots                     |                    12 robots                    |                      16 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/reverse_transport_4_7_LEMURS.gif" height="140"> | <img src="figs/reverse_transport_8_3_LEMURS.gif" height="140"> | <img src="figs/reverse_transport_12_4_LEMURS.gif" height="140"> | <img src="figs/reverse_transport_16_9_LEMURS.gif" height="140"> |  

### Sampling

|                     3 robots                      |                    5 robots                     |                    7 robots                     |                       9 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/sampling_3_0_LEMURS.gif" height="140"> | <img src="figs/sampling_5_3_LEMURS.gif" height="140"> | <img src="figs/sampling_7_0_LEMURS.gif" height="140"> | <img src="figs/sampling_9_0_LEMURS.gif" height="140"> |  

### Navigation

|                     4 robots                      |                    5 robots                     |                    6 robots                     |                       8 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/simple_spread_4_9_LEMURS.gif" height="140"> | <img src="figs/simple_spread_5_0_LEMURS.gif" height="140"> | <img src="figs/simple_spread_6_1_LEMURS.gif" height="140"> | <img src="figs/simple_spread_8_0_LEMURS.gif" height="140"> |  

### Food collection

|                     3 robots                      |                    6 robots                     |                    12 robots                    |                      24 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/simple_spread_food_3_2_LEMURS.gif" height="140"> | <img src="figs/simple_spread_food_6_2_LEMURS.gif" height="140"> | <img src="figs/simple_spread_food_12_1_LEMURS.gif" height="140"> | <img src="figs/simple_spread_food_24_6_LEMURS.gif" height="140"> |  

### Grassland

|                     6 robots                      |                    12 robots                    |                    24 robots                    |                      48 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/grassland_vmas_3_8_LEMURS.gif" height="140"> | <img src="figs/grassland_vmas_6_2_LEMURS.gif" height="140"> | <img src="figs/grassland_vmas_12_1_LEMURS.gif" height="140"> | <img src="figs/grassland_vmas_24_9_LEMURS.gif" height="140"> |  

### Adversarial

|                     6 robots                      |                    12 robots                    |                    24 robots                    |                      48 robots                       |
|:-------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------:|
| <img src="figs/adversarial_vmas_3_2_LEMURS.gif" height="140"> | <img src="figs/adversarial_vmas_6_1_LEMURS.gif" height="140"> | <img src="figs/adversarial_vmas_12_1_LEMURS.gif" height="140"> | <img src="figs/adversarial_vmas_24_0_LEMURS.gif" height="140"> |  


## Code
The code is based on ```gym``` and makes use of some of the scenarios from ```vmas```. 
````sampling```` scenario is reimplemented to fit the version used in the paper. Besides, the 
scenarios with name ````food collection````, ```grassland``` and ```adversaries``` are implementations
of the scenarios used in the paper "DARL1N: ", but parallelized as in the ```vmas``` simulator.

The file ````...```` includes instructions about how to modify ```vmas``` to include 
the ````sampling````, ````food collection````, ```grassland``` and ```adversaries``` scenarios 

We provide the weights of ours policies for the different scenarios in the 
folder ````data/````.

Nevertheless, you can train your own policies by executing ````python training.py````, tuning
the training parameters in the file ````parse_args.py````. You can run ````python evaluation.py````
to evaluate your trained policies and get some cool animations. The evaluation parameters can also be tuned
in the file ````parse_args.py````. The other files are auxiliary, they provide the metrics and pannels displayed
in the paper.

## Citation
If you find our papers/code useful for your research, please cite our work as follows.

E. Sebastian, T. Duong, N. Atanasov, E. Montijano, C. Sagues. [Physics-Informed Multi-Agent Reinforcement Learning for Distributed Multi-Robot Problems](https://eduardosebastianrodriguez.github.io/phMARL/). Under review at IEEE T-RO, 2024

 ```bibtex
@article{sebastian24phMARL,
author = {Eduardo Sebasti\'{a}n AND Thai Duong AND Nikolay Atanasov AND Eduardo Montijano AND Carlos Sag\"{u}\'{e}s},
title = {{Physics-Informed Multi-Agent Reinforcement Learning for Distributed Multi-Robot Problems}},
journal = {arXiv},
pages={1--14},
year = {2024}
}
```
