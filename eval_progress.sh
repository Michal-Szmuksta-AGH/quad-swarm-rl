#!/usr/bin/env bash
# Podglad postepu ewaluacji eval_topology.sh.
#
# Uzycie:
#   ./eval_progress.sh                    # jednorazowy status
#   ./eval_progress.sh -w                 # odswiezanie co 10s (Ctrl+C konczy)
#   LOG=inny.log ./eval_progress.sh       # inny plik logu
#
# JSON-y (eval_metrics_topo_*.json) powstaja dopiero PO zakonczeniu calej
# topologii, wiec ich brak nie oznacza bledu — patrz licznik epizodow ponizej.

LOG="${LOG:-eval_topomix_1B.log}"
N_TOPO="${N_TOPO:-4}"
EPISODES="${EPISODES:-300}"

status() {
  if [[ ! -f "$LOG" ]]; then
    echo "Brak pliku logu: $LOG"
    return 1
  fi

  # Czy proces jeszcze zyje?
  if pgrep -f "swarm_rl.eval_metrics" > /dev/null; then
    state="RUNNING"
  else
    state="STOPPED"
  fi

  sed 's/\x1b\[[0-9;]*m//g' "$LOG" | awk -v ntopo="$N_TOPO" -v eps="$EPISODES" -v state="$state" '
    /  .* @  topology=/ {
      n++
      match($0, /topology=[a-z]+/)
      topo = substr($0, RSTART+9, RLENGTH-9)
      ep = 0
    }
    /Env episodes done/ { split($0, a, "done: "); split(a[2], b, " "); ep = b[1] }
    /^  → / { finished++ }
    END {
      total = ntopo * eps
      done  = (n-1)*eps + ep
      if (done < 0) done = 0
      pct = (total > 0) ? 100*done/total : 0

      bar_w = 40
      filled = int(bar_w * pct / 100)
      bar = ""
      for (i = 0; i < bar_w; i++) bar = bar (i < filled ? "#" : ".")

      printf "  [%s]  %s\n", state, strftime("%H:%M:%S")
      printf "  topologia %d/%d: %-9s   epizody %d/%d\n", n, ntopo, topo, ep, eps
      printf "  [%s] %.1f%%  (%d/%d epizodow, ukonczonych topologii: %d)\n", bar, pct, done, total, finished
    }'

  echo
  echo "  Gotowe JSON-y:"
  local jsons
  jsons=$(ls -1 train_dir/*_1B/eval_metrics_topo_*.json 2>/dev/null)
  if [[ -n "$jsons" ]]; then
    echo "$jsons" | sed 's/^/    /'
  else
    echo "    (jeszcze zadnego)"
  fi
}

if [[ "$1" == "-w" ]]; then
  while true; do
    clear
    status
    sleep 10
  done
else
  status
fi
