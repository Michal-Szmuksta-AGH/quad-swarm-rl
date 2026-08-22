#!/usr/bin/env bash
# Topology cross-evaluation: kazdy model wytrenowany na 'grid' testowany na 4 topologiach.
# Output: train_dir/<model>/eval_metrics_topo_<topology>.json
#
# FAIR COMPARISON: wszystkie modele testowane na TYM SAMYM zestawie EPISODES
# topologii (via TOPOLOGY_SEED=42 + deterministic seed sequence). Bez tego
# roznica miedzy modelami moglaby byc artefaktem roznych sampli topologii.
#
# Customize (env vars):
#   MODELS         — space-separated experiment names (default: 4 gotowe modele)
#   TOPOLOGIES     — default: grid poisson cluster building
#   EPISODES       — default: 100 (per (model, topology))
#   TOPOLOGY_SEED  — default: 42 (base seed dla generacji topologii)
#   TRAIN_DIR, PYTHON — standardowe

set -e

PYTHON="${PYTHON:-$HOME/miniconda3/envs/swarm-rl/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-train_dir}"
EPISODES="${EPISODES:-100}"
# Fixed seed dla topology generation -> wszystkie modele testowane na TYM SAMYM
# zestawie EPISODES topologii (fair comparison, brak variance z topology sampling).
# Kazdy epizod dostaje inny seed (base + episode_idx).
TOPOLOGY_SEED="${TOPOLOGY_SEED:-42}"

MODELS=(${MODELS:-paper_baseline_8drones_s0 perception_limited_r1.0_8drones_s0 perception_limited_r0.2_8drones_s0 multiranger_r4.0_8drones_s0})
TOPOLOGIES=(${TOPOLOGIES:-grid poisson cluster building})

for MODEL in "${MODELS[@]}"; do
  for TOPO in "${TOPOLOGIES[@]}"; do
    TAG="eval_metrics_topo_${TOPO}.json"
    echo ""
    echo "================================================================"
    echo "  $MODEL  @  topology=$TOPO"
    echo "================================================================"

    $PYTHON -m swarm_rl.eval_metrics \
      --algo=APPO --env=quadrotor_multi \
      --train_dir="$TRAIN_DIR" --experiment="$MODEL" \
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
      --quads_obst_topology=$TOPO \
      --quads_topology_seed=$TOPOLOGY_SEED \
      \
      --quads_domain_random=False \
      --quads_obst_density_random=False \
      --quads_obst_size_random=False \
      \
      --replay_buffer_sample_prob=0.0 \
      --with_wandb=False

    mv "$TRAIN_DIR/$MODEL/eval_metrics.json" "$TRAIN_DIR/$MODEL/$TAG"
    echo "  → $TRAIN_DIR/$MODEL/$TAG"
  done
done

# --- Summary ---------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TOPOLOGY EVAL SUMMARY"
echo "================================================================"
printf "%-45s %-10s %-10s %-10s %-10s\n" "Model" "topology" "success" "obst_col" "n_col_obs"
echo "-----------------------------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
  for TOPO in "${TOPOLOGIES[@]}"; do
    TAG="eval_metrics_topo_${TOPO}.json"
    $PYTHON - "$TRAIN_DIR/$MODEL/$TAG" "$MODEL" "$TOPO" <<'PY'
import json, sys
path, model, topo = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
m = d["metrics"]
succ = m["metric/agent_success_rate"]["mean"]
ocol = m["metric/agent_obst_col_rate"]["mean"]
ncol = m["num_collisions_obst_quad"]["mean"]
print(f"{model:<45s} {topo:<10s} {succ:<10.3f} {ocol:<10.3f} {ncol:<10.3f}")
PY
  done
done
echo "================================================================"
