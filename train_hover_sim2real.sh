#!/usr/bin/env bash
# Single-drone hover training dla sim2real deployment na Crazyflie.
# Prosty MLP self_obs → thrusts, bez obstacles, bez neighbors.
# Sim2real konwersja: --model_type=single (parametryczna, znany działający path).
#
# Cel: dron utrzymuje pozycję goal (hover), reaguje na perturbacje.
# NIE avoid obstacles — to jest wersja "proof-of-concept sim2real works".
#
# Usage:
#   SEED=0 bash train_hover_sim2real.sh
#   STEPS=100000000 bash train_hover_sim2real.sh   # smoke test

PYTHON=python
STEPS=${STEPS:-500000000}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-60}
EXPERIMENT=${EXPERIMENT:-hover_sim2real_s${SEED}}

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
  --quads_mode=static_same_goal --quads_episode_duration=15.0 --quads_num_agents=1 \
  --quads_obs_repr=xyz_vxyz_R_omega \
  --quads_encoder_type=mlp \
  --quads_sim2real=True \
  --quads_neighbor_obs_type=none \
  --quads_neighbor_visible_num=0 \
  \
  --quads_collision_reward=0.0 \
  --quads_use_obstacles=False \
  --quads_obstacle_obs_type=none \
  --quads_domain_random=False \
  \
  --quads_use_downwash=False --with_wandb=False \
  --experiment=$EXPERIMENT
