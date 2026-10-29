#!/usr/bin/env bash
# Preview by default. Run manually inside Northflank with --run.
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
svdd_k="${SVDD_K:-10}"
guided_steps="${GUIDED_STEPS:-999}"
late_guided_steps="${LATE_GUIDED_STEPS:-200}"
temperature="${TEMPERATURE:-0}"
python_bin="${GSDIFF_PYTHON:-python}"
config="${EVAL_CONFIG:-test_time/configs/experiment_rplan18.json}"
output_root="${EVAL_OUTPUT:-test_time/outputs/rplan18_sweep_001}"
read -r -a variants <<< "${VARIANTS-baseline best_of_n svdd_pm_last20 svdd_pm_full}"
if (( ${#variants[@]} == 0 )); then
  echo "VARIANTS must contain at least one sweep variant." >&2
  exit 2
fi
seen=" "
for variant in "${variants[@]}"; do
  case "$variant" in
    baseline|best_of_n|svdd_pm_last20|svdd_pm_full) ;;
    *) echo "Unsupported VARIANTS entry: $variant" >&2; exit 2 ;;
  esac
  if [[ "$seen" == *" $variant "* ]]; then
    echo "Duplicate VARIANTS entry: $variant" >&2
    exit 2
  fi
  seen+="$variant "
done

# Match evaluated denoiser states, not batched forward calls or elapsed time.
# SVDD uses T + K*G states per output; K=1 takes the baseline shortcut.
denoising_steps=1000
if [[ -n "${BEST_N:-}" ]]; then
  echo "BEST_N is now calculated from full SVDD. Unset BEST_N to match its denoiser budget." >&2
  exit 2
fi
if [[ ! "$svdd_k" =~ ^[1-9][0-9]*$ ]]; then
  echo "SVDD_K must be a positive integer." >&2
  exit 2
fi
for steps in "$guided_steps" "$late_guided_steps"; do
  if [[ ! "$steps" =~ ^(0|[1-9][0-9]*)$ ]] || (( steps >= denoising_steps )); then
    echo "GUIDED_STEPS and LATE_GUIDED_STEPS must be integers from 0 to 999." >&2
    exit 2
  fi
done
full_states=$((denoising_steps + (svdd_k > 1 ? svdd_k * guided_steps : 0)))
late_states=$((denoising_steps + (svdd_k > 1 ? svdd_k * late_guided_steps : 0)))
# N must be an integer: round to the nearest complete trajectory, ties upward.
best_n=$(((full_states + denoising_steps / 2) / denoising_steps))

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
printf 'Config: %s; selected variants: %s\n' "$config" "${variants[*]}"
printf 'Plan: %s batches x %s trials = %s selected outputs per variant (%s variants).\n' \
  "${#batch_seeds[@]}" "$trials" "$(( ${#batch_seeds[@]} * trials ))" "${#variants[@]}"
printf 'SVDD K=%s, full guidance=%s steps, late guidance=%s steps, temperature=%s.\n' \
  "$svdd_k" "$guided_steps" "$late_guided_steps" "$temperature"
printf 'Target denoiser states per output: full SVDD=%s vs Best-of-%s=%s.\n' \
  "$full_states" "$best_n" "$((best_n * denoising_steps))"
printf 'Late SVDD uses %s states per output and is a separate, unmatched comparison.\n' "$late_states"
printf 'N uses nearest-integer rounding. Reports verify measured counts; edge models, rewards and runtime are not matched.\n'
printf 'Both SVDD variants denoise all 1000 steps; t=0 is deterministic and unguided.\n'
printf 'Generation batch size=%s; maximum neural inference batch size=%s.\n' "$batch_size" "$inference_batch_size"
printf 'Reward backend=%s; final evaluation uses the CPU reference checker.\n' "$reward_backend"
printf 'Output: %s\n' "$output_root"

if [[ "$dry_run" == false ]]; then
  if [[ -e "$output_root" ]]; then
    echo "Output already exists. Set EVAL_OUTPUT to a new path; partial-run resume is not implemented." >&2
    exit 2
  fi
  mkdir -p "$output_root"
  cp -- "$repo_dir/test_time/launch_eval.sh" "$output_root/launcher.sh"
  compute_pairs='[]'
  if [[ "$seen" == *" best_of_n "* && "$seen" == *" svdd_pm_full "* ]]; then
    printf -v compute_pairs '[{"best_of_n":"best_of_n","svdd":"svdd_pm_full","n":%s,"guided_steps":%s,"svdd_states":%s,"best_of_n_states":%s}]' \
      "$best_n" "$guided_steps" "$full_states" "$((best_n * denoising_steps))"
  fi
  cat > "$output_root/compute_budget.json" <<EOF
{
  "metric": "denoiser_evaluated_states_per_selected_output",
  "denoising_steps": $denoising_steps,
  "svdd_k": $svdd_k,
  "rounding": "nearest integer N, ties upward",
  "runtime_matched": false,
  "pairs": $compute_pairs
}
EOF
fi

execute() {
  printf '  '; printf '%q ' "$@"; printf '\n'
  if [[ "$dry_run" == false ]]; then
    "$@"
  fi
}

for ((batch=0; batch<${#batch_seeds[@]}; batch++)); do
  seed="${batch_seeds[batch]}"
  batch_dir="$output_root/batch_$((batch + 1))_seed_${seed}"
  printf '\nBatch %s: paired design seeds %s..%s\n' "$((batch + 1))" "$seed" "$((seed + trials - 1))"
  # Keep the shorter SVDD comparison before full-trajectory guidance.
  for variant in "${variants[@]}"; do
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

printf '\nFor confidence intervals, diversity diagnostics and the sample gallery, run:\n'
printf '  %q -m test_time.analyze --experiment_dir %q\n' "$python_bin" "$output_root"
printf 'Seed batches are analysed separately; cross-batch aggregation is not implemented.\n'
