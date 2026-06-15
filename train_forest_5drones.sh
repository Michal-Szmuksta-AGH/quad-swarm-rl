#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Trening: rój 5 dronów, nawigacja przez las (walce)
# Uruchomienie:  bash train_forest_5drones.sh
# Monitoring:    tensorboard --logdir=train_dir
# ---------------------------------------------------------------------------

# === USTAWIENIA — edytuj tutaj ============================================
PYTHON=/home/michal/miniconda3/envs/swarm-rl/bin/python
EXPERIMENT=forest_5drones_baseline
STEPS=300000000          # budzet krokow srodowiska. Smoke-test: 2000000
NUM_WORKERS=12           # ~ liczba rdzeni CPU (dostroic 10-12)
NUM_AGENTS=5             # liczba dronow w roju
OBST_DENSITY=0.2         # gestosc lasu
OBST_SIZE=0.6            # srednica walca [m] (mniejsza = cienszy pien)
ANNEAL_COLLISION=100000000   # kary kolizji rosna od 0 do pelnej przez tyle krokow
# ===========================================================================

$PYTHON -m swarm_rl.train \
  --env=quadrotor_multi --algo=APPO --train_for_env_steps=$STEPS --use_rnn=False \
  --num_workers=$NUM_WORKERS --num_envs_per_worker=4 \
  --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 --rollout=128 --batch_size=1024 \
  --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --reward_clip=10 \
  --nonlinearity=tanh --policy_initialization=xavier_uniform --actor_critic_share_weights=False \
  --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 \
  --normalize_input=False --normalize_returns=False --rnn_size=256 --with_pbt=False \
  --quads_use_numba=True --save_milestones_sec=3600 \
  --quads_mode=mix --quads_episode_duration=15.0 --quads_num_agents=$NUM_AGENTS \
  --quads_obs_repr=xyz_vxyz_R_omega_floor --quads_encoder_type=attention \
  --quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 \
  --quads_neighbor_obs_type=pos_vel --quads_neighbor_visible_num=4 \
  --quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=4.0 --quads_collision_smooth_max_penalty=4.0 \
  --quads_use_obstacles=True --quads_obstacle_obs_type=octomap \
  --quads_obst_spawn_area 8 8 --quads_obst_density=$OBST_DENSITY --quads_obst_size=$OBST_SIZE \
  --quads_obst_collision_reward=5.0 \
  --anneal_collision_steps=$ANNEAL_COLLISION --replay_buffer_sample_prob=0.75 \
  --quads_use_downwash=True --with_wandb=False \
  --experiment=$EXPERIMENT
