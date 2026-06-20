#!/usr/bin/env bash
# Paper baseline + Multiranger obstacle observation (4 ToF rays vs 9 SDF floats).
# Simulates the Bitcraze Multiranger Deck (4x VL53L1x, 4m range, no FOV cone).
#
# Usage:
#   SEED=0 bash train_multiranger_8drones.sh
#   MAX_RANGE=2.0 NOISE_STD=0.03 bash train_multiranger_8drones.sh   # constrained + noisy
#   STEPS=50000000 bash train_multiranger_8drones.sh                 # smoke test

PYTHON=python
STEPS=${STEPS:-1000000000}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-84}
MAX_RANGE=${MAX_RANGE:-4.0}
NOISE_STD=${NOISE_STD:-0.0}
EXPERIMENT=${EXPERIMENT:-multiranger_r${MAX_RANGE}_8drones_s${SEED}}

$PYTHON -m swarm_rl.train \
  --env=quadrotor_multi --algo=APPO --train_for_env_steps=$STEPS --use_rnn=False \
  --num_workers=$NUM_WORKERS --num_envs_per_worker=4 \
  --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 --rollout=128 --batch_size=1024 \
  --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --reward_clip=10 \
  --nonlinearity=tanh --policy_initialization=xavier_uniform --actor_critic_share_weights=False \
  --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 \
  --normalize_input=False --normalize_returns=False --rnn_size=256 --with_pbt=False \
  --quads_use_numba=True --save_milestones_sec=1800 \
  --seed=$SEED \
  \
  --quads_mode=mix --quads_episode_duration=15.0 --quads_num_agents=8 \
  --quads_obs_repr=xyz_vxyz_R_omega_floor --quads_encoder_type=attention \
  --quads_neighbor_encoder_type=no_encoder --quads_neighbor_hidden_size=256 \
  --quads_neighbor_obs_type=pos_vel \
  --quads_neighbor_visible_num=2 \
  \
  --quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=4.0 --quads_collision_smooth_max_penalty=4.0 \
  --quads_use_obstacles=True \
  --quads_obstacle_obs_type=multiranger \
  --quads_multiranger_max_range=$MAX_RANGE \
  --quads_multiranger_noise_std=$NOISE_STD \
  --quads_multiranger_fov_deg=27.0 \
  --quads_multiranger_num_rays=8 \
  --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 \
  --quads_obst_collision_reward=5.0 \
  --quads_domain_random=False \
  --quads_obst_density_random=False \
  --quads_obst_size_random=False \
  --anneal_collision_steps=300000000 --replay_buffer_sample_prob=0.75 \
  --quads_use_downwash=True --with_wandb=False \
  --experiment=$EXPERIMENT
