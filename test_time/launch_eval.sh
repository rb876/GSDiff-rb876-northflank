#!/usr/bin/env bash
# Preview by default. Run manually inside Northflank with --run after calibration.
set -euo pipefail

case "${1:---dry-run}" in
  --dry-run) dry_run=true ;;
  --run) dry_run=false ;;
  *) echo "Usage: bash test_time/launch_eval.sh [--dry-run|--run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Settings use environment variables; see test_time/README.md." >&2
  exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# One range containing 100 independent design seeds by default.
# A range's first seed is a batch label, not a single seed reused for every design.
read -r -a batch_seeds <<< "${EVAL_SEEDS:-10000}"
trials="${TRIALS_PER_RUN:-100}"
batch_size="${BATCH_SIZE:-4}"
inference_batch_size="${INFERENCE_BATCH_SIZE:-10}"
reward_backend="${REWARD_BACKEND:-gpu}"
best_n="${BEST_N:-2}"
svdd_k="${SVDD_K:-10}"
guided_steps="${GUIDED_STEPS:-999}"
late_guided_steps="${LATE_GUIDED_STEPS:-200}"
temperature="${TEMPERATURE:-0}"
python_bin="${GSDIFF_PYTHON:-python}"
config="${EVAL_CONFIG:-test_time/configs/experiment_v2.json}"
output_root="${EVAL_OUTPUT:-test_time/outputs/evaluation_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$trials" =~ ^[1-9][0-9]*$ ]] || (( ${#batch_seeds[@]} == 0 )); then
  echo "TRIALS_PER_RUN must be positive and EVAL_SEEDS must contain batch start seeds." >&2
  exit 2
fi
for ((i=0; i<${#batch_seeds[@]}; i++)); do
  if [[ ! "${batch_seeds[i]}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "EVAL_SEEDS must contain nonnegative integers without leading zeros." >&2
    exit 2
  fi
  for ((j=0; j<i; j++)); do
    if (( batch_seeds[i] < batch_seeds[j] + trials && batch_seeds[j] < batch_seeds[i] + trials )); then
      echo "Batch seed ranges overlap; separate their starts by at least TRIALS_PER_RUN." >&2
      exit 2
    fi
  done
done

printf 'Mode: %s\n' "$([[ "$dry_run" == true ]] && echo preview || echo run)"
printf 'Plan: %s batches x %s trials = %s selected outputs per variant (four variants).\n' \
  "${#batch_seeds[@]}" "$trials" "$(( ${#batch_seeds[@]} * trials ))"
printf 'Best-of-%s; SVDD K=%s, full guidance=%s steps, late guidance=%s steps, temperature=%s.\n' \
  "$best_n" "$svdd_k" "$guided_steps" "$late_guided_steps" "$temperature"
printf 'Both SVDD variants denoise all 1000 steps; t=0 is deterministic and unguided.\n'
printf 'Generation batch size=%s; maximum neural inference batch size=%s.\n' "$batch_size" "$inference_batch_size"
printf 'Reward backend=%s; final evaluation uses the CPU reference checker.\n' "$reward_backend"
printf 'Compute matching is provisional until calibrated. Output: %s\n' "$output_root"

if [[ "$dry_run" == false ]]; then
  if [[ -e "$output_root" ]]; then
    echo "Output already exists. Set EVAL_OUTPUT to a new path; partial-run resume is not implemented." >&2
    exit 2
  fi
  mkdir -p "$output_root"
  cp -- "$repo_dir/test_time/launch_eval.sh" "$output_root/launcher.sh"
fi

execute() {
  printf '  '; printf '%q ' "$@"; printf '\n'
  if [[ "$dry_run" == false ]]; then
    "$@"
  fi
}

variants=(baseline best_of_n svdd_pm_full svdd_pm_last20)
for ((batch=0; batch<${#batch_seeds[@]}; batch++)); do
  seed="${batch_seeds[batch]}"
  batch_dir="$output_root/batch_$((batch + 1))_seed_${seed}"
  printf '\nBatch %s: paired design seeds %s..%s\n' "$((batch + 1))" "$seed" "$((seed + trials - 1))"
  # Run serially on the same GPU; rotate variant order across batches.
  for ((offset=0; offset<${#variants[@]}; offset++)); do
    variant="${variants[(batch + offset) % ${#variants[@]}]}"
    method="$variant"
    extra=()
    case "$variant" in
      best_of_n) extra=(--n "$best_n") ;;
      svdd_pm_full)
        method=svdd_pm
        extra=(--k "$svdd_k" --guided_steps "$guided_steps" --temperature "$temperature") ;;
      svdd_pm_last20)
        method=svdd_pm
        extra=(--k "$svdd_k" --guided_steps "$late_guided_steps" --temperature "$temperature") ;;
    esac
    command=("$python_bin" -u -m test_time.run --config "$config" --method "$method"
             --trials "$trials" --seed "$seed" --batch_size "$batch_size"
             --inference_batch_size "$inference_batch_size" --reward_backend "$reward_backend"
             --output_dir "$batch_dir/$variant" "${extra[@]}")
    if [[ "$dry_run" == false ]]; then
      mkdir -p "$batch_dir"
      execute "${command[@]}" 2>&1 | tee "$batch_dir/$variant.log"
    else
      execute "${command[@]}"
    fi
  done
  # Compare only paired runs within this batch. Cross-batch pooling is still pending.
  execute "$python_bin" -m test_time.evaluate --experiment_dir "$batch_dir"
done

printf '\nCross-batch aggregation, confidence intervals and dedicated diversity metrics are not yet implemented.\n'
