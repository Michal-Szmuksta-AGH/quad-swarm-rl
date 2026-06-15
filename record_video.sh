#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wysokiej jakości nagrania modelu z HUD-em metryk:
#   * Renderowanie 2560x1440 → downscale ffmpegiem do 1920x1080 (naturalny AA)
#   * Overlay tekstu: licznik epizodów, czas, kolizje dron-dron, kolizje
#     dron-przeszkoda, drony przy celu
#   * Trajektorie 100 klatek długości (4x dłuższe niż domyślne 25)
#   * Optional slow-motion: SLOWMO=2 spowalnia 2× przez ffmpeg setpts
#   * 6 filmów = 2 scenariusze x 3 perspektywy
#
# Wymagania: ffmpeg (sudo apt install ffmpeg)
# Output:   train_dir/<EXPERIMENT>/videos/<scenariusz>_<kamera>.mp4
#
# Czas: ~6 x (EPISODES x 15s rendering @ 2K + encode + downscale) = ~15-20 min
# dla 3 epizodów na film na RTX 2060.
#
# Usage:
#   bash record_video.sh                          # 5 epizodów, paper_baseline_8drones_s0
#   EPISODES=3 bash record_video.sh               # krótsze filmy
#   SLOWMO=2 bash record_video.sh                 # 2x wolniejsze odtwarzanie
#   EXPERIMENT=forest_8drones_dr bash record_video.sh
# ---------------------------------------------------------------------------

set -e

PYTHON="${PYTHON:-$HOME/miniconda3/envs/swarm-rl/bin/python}"
EXPERIMENT="${EXPERIMENT:-paper_baseline_8drones_s0}"
TRAIN_DIR="${TRAIN_DIR:-train_dir}"
EPISODES="${EPISODES:-5}"
FPS="${FPS:-30}"
SLOWMO="${SLOWMO:-1}"   # 1 = normal, 2 = polowa szybkosci, 3 = jedna trzecia itd.

SCENARIOS=(o_random o_static_same_goal)
VIEWS=(global topdown chase)

OUT_DIR="$TRAIN_DIR/$EXPERIMENT/videos"
TMP_DIR="$TRAIN_DIR/$EXPERIMENT/videos/_tmp_raw"
mkdir -p "$OUT_DIR" "$TMP_DIR"

echo "=== High-quality video recording with HUD ==="
echo "Experiment:   $EXPERIMENT"
echo "Scenariusze:  ${SCENARIOS[*]}"
echo "Kamery:       ${VIEWS[*]}"
echo "Epizody/film: $EPISODES"
echo "FPS:          $FPS"
echo "Slow-motion:  ${SLOWMO}x"
echo "Output:       $OUT_DIR/"
echo ""

if ! command -v ffmpeg &>/dev/null; then
  echo "BLAD: ffmpeg nie jest zainstalowany"
  echo "      sudo apt install ffmpeg"
  exit 1
fi

TOTAL=$(( ${#SCENARIOS[@]} * ${#VIEWS[@]} ))
COUNT=0

for SCENARIO in "${SCENARIOS[@]}"; do
  for VIEW in "${VIEWS[@]}"; do
    COUNT=$((COUNT + 1))
    BASE_NAME="${SCENARIO}_${VIEW}"
    RAW_PATH="$TMP_DIR/${BASE_NAME}_raw.mp4"
    FINAL_PATH="$OUT_DIR/${BASE_NAME}.mp4"

    echo "[$COUNT/$TOTAL] $BASE_NAME"
    echo "    1/2 Renderowanie 2K + HUD ..."

    $PYTHON -m swarm_rl.record_overlay \
      --algo=APPO --env=quadrotor_multi \
      --train_dir="$TRAIN_DIR" --experiment="$EXPERIMENT" \
      --device=gpu \
      --video_output="$RAW_PATH" \
      --fps=$FPS \
      --max_num_episodes=$EPISODES \
      --eval_deterministic=False \
      \
      --quads_render=True \
      --quads_view_mode $VIEW \
      --quads_use_numba=False \
      \
      --quads_num_agents=8 \
      --quads_episode_duration=15.0 \
      --quads_mode=$SCENARIO \
      --quads_room_dims 10 10 10 \
      --quads_use_downwash=True \
      \
      --quads_use_obstacles=True \
      --quads_obstacle_obs_type=octomap \
      --quads_obst_spawn_area 8 8 \
      --quads_obst_density=0.2 \
      --quads_obst_size=0.6 \
      \
      --quads_domain_random=False \
      --quads_obst_density_random=False \
      --quads_obst_size_random=False \
      \
      --replay_buffer_sample_prob=0.0 \
      --with_wandb=False \
      2>&1 | tail -3

    if [ ! -f "$RAW_PATH" ]; then
      echo "    BLAD: nie znaleziono $RAW_PATH"
      continue
    fi

    # Krok 2: ffmpeg downscale 2K -> 1080p (anti-aliasing przez lanczos) + opcjonalny slow-mo
    echo "    2/2 Downscale 2K->1080p + ${SLOWMO}x slow-mo, encode H.264 ..."

    if [ "$SLOWMO" = "1" ]; then
      VFILTER="scale=1920:1080:flags=lanczos"
    else
      VFILTER="scale=1920:1080:flags=lanczos,setpts=${SLOWMO}.0*PTS"
    fi

    ffmpeg -y -loglevel error -i "$RAW_PATH" \
      -vf "$VFILTER" \
      -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
      "$FINAL_PATH"

    if [ -f "$FINAL_PATH" ]; then
      SIZE=$(du -h "$FINAL_PATH" | cut -f1)
      echo "    OK: $FINAL_PATH ($SIZE)"
    fi
    echo ""
  done
done

# Sprzatanie surowych plików (zostaw jeśli chcesz mieć backup w 2K bez HUD overlay zmian)
rm -rf "$TMP_DIR"

echo "=== Gotowe ==="
ls -lh "$OUT_DIR/"
