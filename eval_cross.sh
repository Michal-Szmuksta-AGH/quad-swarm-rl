#!/usr/bin/env bash
# Cross-evaluation: each trained model evaluated at multiple sensor ranges.
# Tells us whether train-time perception constraint matters, or whether any
# policy is naturally robust to test-time range limiting.
#
# Customize via env vars:
#   MODELS  — space-separated experiment names    (default: paper_baseline + r=1.0)
#   RANGES  — space-separated sensor ranges       (default: 100.0 1.0)
#   EPISODES, TRAIN_DIR, PYTHON                  — standard overrides
#
# Output: train_dir/<model>/eval_metrics_r<range>.json (per combination)
# Plus summary table printed at the end.

set -e

PYTHON="${PYTHON:-$HOME/miniconda3/envs/swarm-rl/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-train_dir}"
EPISODES="${EPISODES:-50}"

MODELS=(${MODELS:-paper_baseline_8drones_s0 perception_limited_r1.0_8drones_s0})
RANGES=(${RANGES:-100.0 1.0})

for MODEL in "${MODELS[@]}"; do
  for R in "${RANGES[@]}"; do
    TAG="eval_metrics_r${R}.json"
    echo ""
    echo "================================================================"
    echo "  $MODEL  @  sensor_range=$R"
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
      --quads_obst_sensor_range=$R \
      \
      --quads_domain_random=False \
      --quads_obst_density_random=False \
      --quads_obst_size_random=False \
      \
      --replay_buffer_sample_prob=0.0 \
      --with_wandb=False

    # eval_metrics.py always writes to eval_metrics.json; tag it per-range
    mv "$TRAIN_DIR/$MODEL/eval_metrics.json" "$TRAIN_DIR/$MODEL/$TAG"
    echo "  → $TRAIN_DIR/$MODEL/$TAG"
  done
done

# --- Summary table ---------------------------------------------------------
echo ""
echo "================================================================"
echo "  CROSS-EVAL SUMMARY"
echo "================================================================"
printf "%-45s %-10s %-10s %-10s %-10s\n" "Model" "test_r" "success" "obst_col" "n_col_obs"
echo "----------------------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
  for R in "${RANGES[@]}"; do
    TAG="eval_metrics_r${R}.json"
    $PYTHON - "$TRAIN_DIR/$MODEL/$TAG" "$MODEL" "$R" <<'PY'
import json, sys
path, model, r = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
m = d["metrics"]
succ = m["metric/agent_success_rate"]["mean"]
ocol = m["metric/agent_obst_col_rate"]["mean"]
ncol = m["num_collisions_obst_quad"]["mean"]
print(f"{model:<45s} {r:<10s} {succ:<10.3f} {ocol:<10.3f} {ncol:<10.3f}")
PY
  done
done
echo "================================================================"
