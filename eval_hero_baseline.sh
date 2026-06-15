#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Headless eval of a trained hero model in PAPER-MATCHED conditions.
#
# Works with both the original hero (forest_8drones_dr) and the paper-aligned
# rerun (paper_baseline_8drones) checkpoints.  Architecture flags (visible_num,
# encoder type, etc.) are NOT overridden -- they are loaded from each
# checkpoint's saved config.json by the eval_metrics script.  We only override
# environment conditions to make evaluation deterministic and paper-comparable.
#
#   Paper target (Table I, 8 drones, 20% density):
#     success_rate ~ 0.97,  collision_rate ~ 0.03
#
# Usage:
#   bash eval_hero_baseline.sh                                  # original hero
#   EXPERIMENT=paper_baseline_8drones bash eval_hero_baseline.sh   # paper rerun
#   EPISODES=100 bash eval_hero_baseline.sh                     # tighter CI
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-$HOME/miniconda3/envs/swarm-rl/bin/python}"
EXPERIMENT="${EXPERIMENT:-forest_8drones_dr}"
TRAIN_DIR="${TRAIN_DIR:-train_dir}"
EPISODES="${EPISODES:-50}"

# --- Environment conditions: pin density/size, turn off DR for reproducibility
# Architecture details (num neighbors, encoder type) come from the saved
# config.json of the experiment -- do NOT override them here, or the model
# weights will not match the rebuilt architecture.

$PYTHON -m swarm_rl.eval_metrics \
  --algo=APPO --env=quadrotor_multi \
  --train_dir="$TRAIN_DIR" --experiment="$EXPERIMENT" \
  --device=gpu \
  --max_num_episodes=$EPISODES \
  --eval_deterministic=False \
  --no_render \
  --quads_render=False \
  --quads_use_numba=True \
  \
  --quads_episode_duration=15.0 \
  --quads_mode=mix \
  --quads_room_dims 10 10 10 \
  \
  --quads_use_obstacles=True \
  --quads_obst_spawn_area 8 8 \
  --quads_obst_density=0.2 \
  --quads_obst_size=0.6 \
  \
  --quads_domain_random=False \
  --quads_obst_density_random=False \
  --quads_obst_size_random=False \
  \
  --replay_buffer_sample_prob=0.0 \
  --with_wandb=False
