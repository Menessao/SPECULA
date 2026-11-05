#!/bin/bash
# Offline (detached) run of calibration.py steps, split in shards over GPUs; the processes survive the terminal / session.
#   bash run_offline.sh "pc" "0 1 1"  [extra calibration.py options, e.g. --seeings 0.7 0.9]
#   arg 1: steps; arg 2: GPU of every shard (here 3 shards: one on GPU 0, two on GPU 1)
# Logs: <calib_dir>/logs/offline_<steps>_<k>of<m>.log (one SPECULA log per case next to them). Stop: kill the PIDs printed below.
# Resume: run the same command again (finished cases are skipped). /raid1 is mounted noexec: call it through bash.
set -e
STEPS="$1"; GPUS=($2); shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/home/mmenessini/.conda/envs/specula/bin/python}"
LOGDIR="${CALIB_DIR:-/raid1/mmenessini/calibration/WFSW}/logs"
mkdir -p "$LOGDIR"
M=${#GPUS[@]}
for K in $(seq 0 $((M - 1))); do
    LOG="$LOGDIR/offline_${STEPS// /_}_${K}of${M}.log"
    setsid nohup "$PY" "$HERE/calibration.py" $STEPS --gpu "${GPUS[$K]}" --shard "$K/$M" "$@" > "$LOG" 2>&1 < /dev/null &
    echo "shard $K/$M on GPU ${GPUS[$K]}: PID $!  log $LOG"
done
