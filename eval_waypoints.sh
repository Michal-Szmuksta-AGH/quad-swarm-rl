#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ewaluacja interaktywna: rojowi zadajesz kolejne waypointy klawiszem ENTER.
#
# Domyslny model: paper_baseline_8drones_s0 (paper-aligned, K=2 sasiadow,
# trenowany w 10x10x10 m, density 0.2, size 0.6, BEZ domain randomization).
# Dla starego hero: EXPERIMENT=forest_8drones_dr bash eval_waypoints.sh
#
# UWAGA o dystrybucji:
#   Model byl trenowany w pokoju 10x10x10 ze spawn area 8x8.
#   Tutaj ewaluujemy w 16x16x8 ze spawn area 12x12 zeby miec wiecej miejsca
#   na waypointy (lista trafia do +-5m). To jest OUT OF DISTRIBUTION
#   wzgledem treningu - polityka moze sie zachowywac dziwniej niz na evalu
#   in-distribution. Architektura sieci (visible_num, encoder_type) ladowana
#   z saved config eksperymentu - NIE nadpisywaj w CLI.
#
# Sterowanie w oknie renderowania:
#   ENTER   - kolejny waypoint
#   R       - reset do pierwszego
#   G       - kamera globalna (sferyczna)
#   strzalki - obrot kamery (po G)
#   Z / X   - zoom in / out
#   0-9     - kamera doczepiona do drona o tym indeksie
#   L       - chase za dronem 0
#
# Edycja trasy: gym_art/quadrotor_multi/scenarios/obstacles/o_waypoints.py
#               (lista DEFAULT_WAYPOINTS)
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-$HOME/miniconda3/envs/swarm-rl/bin/python}"
EXPERIMENT="${EXPERIMENT:-paper_baseline_8drones_s0}"
TRAIN_DIR="${TRAIN_DIR:-train_dir}"

$PYTHON -m swarm_rl.enjoy \
  --algo=APPO --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True \
  --quads_mode=o_waypoints \
  --quads_use_obstacles=True --quads_obstacle_obs_type=octomap \
  --quads_room_dims 16 16 8 \
  --quads_obst_spawn_area 12 12 \
  --quads_obst_density=0.2 --quads_obst_size=0.6 \
  --quads_episode_duration=600 \
  --quads_view_mode global \
  --max_num_episodes=999 \
  --train_dir="$TRAIN_DIR" --experiment="$EXPERIMENT"
