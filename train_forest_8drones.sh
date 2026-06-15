#!/usr/bin/env bash
PYTHON=python
STEPS=${STEPS:-1000000000}
EXPERIMENT=${EXPERIMENT:-forest_8drones_dr}
NUM_WORKERS=96

$PYTHON -m swarm_rl.train \
  --env=quadrotor_multi --algo=APPO --train_for_env_steps=$STEPS --use_rnn=False \
  --num_workers=$NUM_WORKERS --num_envs_per_worker=4 \
  --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 --rollout=128 --batch_size=1024 \
  --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --reward_clip=10 \
  --nonlinearity=tanh --policy_initialization=xavier_uniform --actor_critic_share_weights=False \
  --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 \
  --normalize_input=False --normalize_returns=False --rnn_size=256 --with_pbt=False \
  --quads_use_numba=True --save_milestones_sec=1800 \
  --quads_mode=mix --quads_episode_duration=15.0 --quads_num_agents=8 \
  --quads_obs_repr=xyz_vxyz_R_omega_floor --quads_encoder_type=attention \
  --quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 \
  --quads_neighbor_obs_type=pos_vel --quads_neighbor_visible_num=6 \
  --quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=4.0 --quads_collision_smooth_max_penalty=4.0 \
  --quads_use_obstacles=True --quads_obstacle_obs_type=octomap \
  --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 \
  --quads_obst_collision_reward=5.0 \
  --quads_domain_random=True \
  --quads_obst_density_random=True --quads_obst_density_min=0.05 --quads_obst_density_max=0.2 \
  --quads_obst_size_random=True --quads_obst_size_min=0.3 --quads_obst_size_max=0.6 \
  --anneal_collision_steps=100000000 --replay_buffer_sample_prob=0.75 \
  --quads_use_downwash=True --with_wandb=False \
  --experiment=forest_8drones_dr
