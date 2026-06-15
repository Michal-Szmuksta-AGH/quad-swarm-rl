#!/usr/bin/env bash
# Paper baseline + SDF range cutoff via --quads_obst_sensor_range.
# SDF values > SENSOR_RANGE are clipped to SENSOR_RANGE (simulates finite-range ToF).
#
# Usage:
#   SENSOR_RANGE=1.0 SEED=0 bash train_perception_limited_8drones.sh
#   STEPS=50000000 bash train_perception_limited_8drones.sh   # smoke test

PYTHON=python
STEPS=${STEPS:-1000000000}
SENSOR_RANGE=${SENSOR_RANGE:-1.0}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-84}
EXPERIMENT=${EXPERIMENT:-perception_limited_r${SENSOR_RANGE}_8drones_s${SEED}}

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
  --quads_use_obstacles=True --quads_obstacle_obs_type=octomap \
  --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 \
  --quads_obst_sensor_range=$SENSOR_RANGE \
  --quads_obst_collision_reward=5.0 \
  --quads_domain_random=False \
  --quads_obst_density_random=False \
  --quads_obst_size_random=False \
  --anneal_collision_steps=300000000 --replay_buffer_sample_prob=0.75 \
  --quads_use_downwash=True --with_wandb=False \
  --experiment=$EXPERIMENT
