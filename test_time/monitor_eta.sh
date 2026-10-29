#!/usr/bin/env bash
# Rough ETA from the last ten completed-trial log timings. Read-only.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${1:-$repo_dir/test_time/outputs/bathroom_001/logs}"
longest=0
waiting=0

duration() {
  awk -v seconds="$1" 'BEGIN { minutes=int((seconds+59)/60); printf "%dh %02dm", int(minutes/60), minutes%60 }'
}

printf '%-18s %12s %12s %14s\n' 'Run' 'Completed' 'Sec/output' 'Remaining ~'
for method in svdd_last20 svdd_full repair hard_mc; do
  file="$log_dir/$method.log"
  if [[ ! -f "$file" ]]; then
    printf '%-18s %s\n' "$method" 'log not found'
    waiting=1
    continue
  fi
  read -r done total average remaining < <(
    awk -v RS='[\r\n]' '
      / trial [0-9]+\/[0-9]+:/ {
        seconds=-1
        for (i=1; i<=NF; i++) {
          if ($i=="trial") { split($(i+1), progress, /[\/:]/); done=progress[1]; total=progress[2] }
          if ($i ~ /^(amortized_seconds|seconds)=/) { split($i, value, "="); seconds=value[2]+0 }
        }
        if (seconds>=0) { timings[count%10]=seconds; count++ }
      }
      END {
        if (!count) { print "0 0 0 -1"; exit }
        size=count<10?count:10
        for (i=0; i<size; i++) sum+=timings[i]
        average=sum/size
        remaining=average*(total-done)
        printf "%d %d %.2f %.0f\n", done,total,average,(remaining>0?remaining:0)
      }' "$file"
  )
  if (( remaining < 0 )); then
    printf '%-18s %s\n' "$method" 'waiting for first completed trial'
    waiting=1
  else
    printf '%-18s %12s %12s %14s\n' "$method" "$done/$total" "$average" "$(duration "$remaining")"
    if (( remaining > longest )); then longest=$remaining; fi
  fi
done
if (( waiting )); then
  printf '\nOverall ETA: waiting for timing data from all four runs.\n'
else
  printf '\nAll four finished in approximately %s (slowest remaining run).\n' "$(duration "$longest")"
fi
printf 'Rough forecast at recent speed; excludes export overhead and progress within the current trial/batch.\n'
printf 'Assumes jobs remain running; changing GPU contention or variable MC trials can shift the estimate.\n'
