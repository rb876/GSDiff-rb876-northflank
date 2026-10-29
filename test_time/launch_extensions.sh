#!/usr/bin/env bash
# Preview by default; --run explicitly launches the selected extensions.
set -euo pipefail

dry_run=true
smoke=false
for argument in "$@"; do
  case "$argument" in
    --dry-run) dry_run=true ;;
    --run) dry_run=false ;;
    --smoke) smoke=true ;;
    --help|-h)
      echo "Usage: bash test_time/launch_extensions.sh [--smoke] [--dry-run|--run]"
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON:-python}"
trials="${TRIALS_PER_RUN:-100}"
seed="${EVAL_SEED:-10000}"
inference_batch_size="${INFERENCE_BATCH_SIZE:-10}"
device="${DEVICE:-cuda:0}"
output_root="${EVAL_OUTPUT:-test_time/outputs/extensions_001}"
read -r -a methods <<< "${METHODS:-hard_h_mc svdd_repair_aware}"
if [[ "$smoke" == true ]]; then
  trials=2
fi

if [[ ! "$trials" =~ ^[1-9][0-9]*$ || ! "$inference_batch_size" =~ ^[1-9][0-9]*$ ||
      ! "$seed" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "TRIALS_PER_RUN and INFERENCE_BATCH_SIZE must be positive integers; EVAL_SEED must be nonnegative." >&2
  exit 2
fi
if (( ${#methods[@]} == 0 )); then
  echo "METHODS must contain hard_h_mc and/or svdd_repair_aware." >&2
  exit 2
fi
seen=" "
for method in "${methods[@]}"; do
  case "$method" in
    hard_h_mc|svdd_repair_aware) ;;
    *) echo "Unsupported method: $method" >&2; exit 2 ;;
  esac
  if [[ "$seen" == *" $method "* ]]; then
    echo "Duplicate method: $method" >&2
    exit 2
  fi
  seen+="$method "
done

printf 'Mode: %s; smoke: %s\n' "$([[ "$dry_run" == true ]] && echo preview || echo run)" "$smoke"
printf 'Trials per method: %s; first design seed: %s; inference batch: %s; device: %s\n' \
  "$trials" "$seed" "$inference_batch_size" "$device"
printf 'Config: test_time/configs/extensions_v1.json; output: %s\n' "$output_root"
printf 'Last-20%% settings and final repair come from the saved extension config.\n'

# Check every selected destination before starting either method.
if [[ "$dry_run" == false ]]; then
  for method in "${methods[@]}"; do
    for path in "$output_root/$method" "$output_root/$method.log"; do
      if [[ -e "$path" || -L "$path" ]]; then
        echo "Output already exists: $path. Choose a new EVAL_OUTPUT; overwrite/resume is not supported." >&2
        exit 2
      fi
    done
  done
  mkdir -p -- "$output_root"
fi

execute() {
  local log_path="$1"
  shift
  printf '  '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" == false ]]; then
    "$@" 2>&1 | tee -- "$log_path"
  fi
}

for method in "${methods[@]}"; do
  extra=()
  if [[ "$smoke" == true ]]; then
    extra=(--k 2)
    if [[ "$method" == hard_h_mc ]]; then
      extra+=(--rollouts 2 --terminal_k 2)
    fi
  fi
  execute "$output_root/$method.log" "$python_bin" -u -m test_time.run_extensions \
    --method "$method" --config test_time/configs/extensions_v1.json \
    --trials "$trials" --seed "$seed" --inference_batch_size "$inference_batch_size" \
    --device "$device" --output_dir "$output_root/$method" "${extra[@]}"
done

execute "$output_root/comparison.log" "$python_bin" -m test_time.run_extensions \
  --summarize_root "$output_root"
