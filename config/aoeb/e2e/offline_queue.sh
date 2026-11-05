#!/bin/bash
# One-off offline queue of the e2e calibration (call through bash; log: <calib>/logs/offline_queue.log):
#  1. wait until the main_simul notebook loop has stopped (no SPECULA with temp_overrides.yml and no new aliasing file for 10 min)
#  2. migrate the WFSW files to the aoeb names (also deletes the files of the rate scan)
#  3. aliasing runs at the new seeings (seeing 0.9 comes from the migrated files)
#  4. wait for the PC jobs at 0.9 already running, then PC IMs at the new seeings. Finished cases are skipped: safe to run again.
HERE="$(cd "$(dirname "$0")" && pwd)"; PY=/home/mmenessini/.conda/envs/specula/bin/python; W=/raid1/mmenessini/calibration/WFSW
SEEINGS="0.6 0.75 1.05"
GPUS=(1 1 0)
say() { echo "$(date '+%F %T') $*"; }
shards() {          # step: one process per entry of GPUS, waits for all of them
    local pids=() rc=0
    for K in 0 1 2; do "$PY" "$HERE/calibration.py" "$1" --seeings $SEEINGS --gpu "${GPUS[$K]}" --shard "$K/3" > "$W/logs/offline_$1_new_${K}of3.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}
idle=0
while [ $idle -lt 10 ]; do
    if pgrep -f "calib_wfs_aliasing.yml temp_overrides.yml" > /dev/null || [ -n "$(find $W/aliasing -name '*Hz_alias.fits' -mmin -10 -print -quit)" ]; then idle=0; else idle=$((idle + 1)); fi
    sleep 60
done
say "notebook loop stopped: migrating"
"$PY" "$HERE/migrate_names.py" --apply || { say "MIGRATION FAILED"; exit 1; }
say "aliasing at $SEEINGS"
shards alias || { say "ALIAS FAILED: see logs/offline_alias_new_*"; exit 1; }
say "waiting for the PC jobs at 0.9"
while pgrep -f "calibration.py pc .*--seeings 0.9" > /dev/null; do sleep 300; done
say "PC IMs at $SEEINGS"
shards pc || { say "PC FAILED: see logs/offline_pc_new_*"; exit 1; }
say "all done"
