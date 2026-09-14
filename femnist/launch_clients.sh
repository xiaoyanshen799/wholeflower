#!/usr/bin/env bash
set -euo pipefail

# Managed profiling keeps its own units, logs, barrier and failure handling.
if [[ -n "${WARMUP_JOB:-}" ]]; then
  WARMUP_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  exec "${WARMUP_PYTHON:-python3}" "$WARMUP_PROJECT_DIR/warmup_launch.py" --job "$WARMUP_JOB"
fi

# 使用方式：
#   PY=/path/to/python ./launch_clients.sh [DATA_DIR] [SERVER_ADDR] [MAX_CLIENTS]
#
# 例子：
#   PY=$(which python3) ./launch_clients.sh data_partitions 172.31.17.220:8081 42

DATA_DIR=${1:-data_partitions}
SERVER=${2:-"127.0.0.1:8081"}
MAX_CLIENTS=${3:-4}   # 默认仅启动前 4 个；设为 0 启动目录里所有 client_*.npz
DATASET=${DATASET:-speech_commands}
MODEL=${MODEL:-resnet34}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.001}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-}
NUM_CLASSES=${NUM_CLASSES:-10}
CPU_ONLY=${CPU_ONLY:-0}
MPS_ENABLE=${MPS_ENABLE:-1}
if [[ "$CPU_ONLY" == "1" ]]; then
  MPS_ENABLE=0
fi
MPS_PERCENT=${MPS_PERCENT:-5}
MPS_PIPE_DIRECTORY=${MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
MPS_LOG_DIRECTORY=${MPS_LOG_DIRECTORY:-/tmp/nvidia-mps}
BIND_CLIENT_TO_CPU=${BIND_CLIENT_TO_CPU:-0}
ENABLE_CPU_AFFINITY=${ENABLE_CPU_AFFINITY:-$BIND_CLIENT_TO_CPU}
CPU_AFFINITY_OFFSET=${CPU_AFFINITY_OFFSET:-0}
CPU_AFFINITY_WRAP=${CPU_AFFINITY_WRAP:-1}
MEASUREMENT_RUN_ID=${MEASUREMENT_RUN_ID:-}
PACER_ENABLE=${PACER_ENABLE:-0}
PACER_LOG_DIR=${PACER_LOG_DIR:-}
PACER_RESOURCE_STATE_DIR=${PACER_RESOURCE_STATE_DIR:-}

PY=${PY:-python3}
# Accept common wrappers like "( python3 )" and optional args like "python3 -u".
PY=$(echo "$PY" | sed -E 's/^[[:space:]]*\((.*)\)[[:space:]]*$/\1/' | xargs)
read -r -a PY_CMD <<< "$PY"
if [[ ${#PY_CMD[@]} -eq 0 ]]; then
  echo "Invalid PY setting: empty value"
  exit 1
fi
if ! command -v "${PY_CMD[0]}" >/dev/null 2>&1; then
  echo "Invalid PY setting: '${PY_CMD[0]}' not found in PATH"
  echo "Example: PY=/home/xiaoyan/wholeflower/fedavgm/venv/bin/python ./launch_clients.sh ..."
  exit 1
fi

# 运行哪个用户（默认当前用户）；之前写死 ubuntu 会导致 217/USER，客户端根本没起来
RUN_AS_USER=${RUN_AS_USER:-"$(id -un)"}

# 强制使用 systemd system + scope 模式（与旧的 --scope 行为一致）
SYSTEMD_MODE="system"

# femnist 项目目录（脚本所在目录），避免从别的路径执行导致 working-directory/data-dir 错乱
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT_DIR/logs"
if [[ -n "${CLIENT_LOG_DIR:-}" ]]; then
  mkdir -p -- "$CLIENT_LOG_DIR"
  CLIENT_LOG_DIR=$(readlink -f "$CLIENT_LOG_DIR")
else
  CLIENT_LOG_DIR=$(mktemp -d "$PROJECT_DIR/logs/clients_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
printf 'client_id,cpu,cpu_affinity\n' > "$CLIENT_LOG_DIR/cpu_config.csv"

# Optional: CSV mapping client_id -> new_cpu (fraction 0-1). If set, use per-client CPUQuota.
CPU_MAP_CSV=${CPU_MAP_CSV:-"$PROJECT_DIR/../logs/client_num_examples.csv"}
CPU_MAP_ONLY=${CPU_MAP_ONLY:-0}
if [[ "$CPU_MAP_ONLY" == "1" && ! -f "$CPU_MAP_CSV" ]]; then
  echo "ERROR: CPU_MAP_ONLY=1 requires an existing CPU_MAP_CSV"
  exit 1
fi

# 把 data_dir 变成绝对路径，避免工作目录不同导致找不到
if [[ "$DATA_DIR" = /* ]]; then
  DATA_DIR_ABS=$(readlink -f "$DATA_DIR")
else
  DATA_DIR_ABS=$(readlink -f "$PROJECT_DIR/$DATA_DIR")
fi

ENV_VARS=(
  --setenv=OMP_NUM_THREADS=1
  --setenv=OPENBLAS_NUM_THREADS=1
  --setenv=MKL_NUM_THREADS=1
  --setenv=NUMEXPR_NUM_THREADS=1
  --setenv=TF_NUM_INTRAOP_THREADS=1
  --setenv=TF_NUM_INTEROP_THREADS=1
  --setenv=CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MPS_PERCENT"
  --setenv=CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY"
  --setenv=CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIRECTORY"
)
if [[ "$CPU_ONLY" == "1" ]]; then
  ENV_VARS+=(--setenv=CUDA_VISIBLE_DEVICES= --setenv=SKIP_TF_GPU_MEMORY_GROWTH=1)
fi

echo "Data dir    : $DATA_DIR_ABS"
echo "Server      : $SERVER"
echo "Max clients : ${MAX_CLIENTS}"
echo "Python      : ${PY_CMD[*]}"
echo "Run as user : $RUN_AS_USER"
echo "Project dir : $PROJECT_DIR"
echo "CPU map CSV : $CPU_MAP_CSV"
echo "Run logs    : $CLIENT_LOG_DIR"
echo "CPU affinity: $ENABLE_CPU_AFFINITY"
echo "Bind cid CPU: $BIND_CLIENT_TO_CPU (offset=$CPU_AFFINITY_OFFSET wrap=$CPU_AFFINITY_WRAP)"
echo "Measure id  : ${MEASUREMENT_RUN_ID:-auto}"
echo "Pacer       : $PACER_ENABLE"
echo "LR          : $LR"
echo "MPS enable  : $MPS_ENABLE"
echo "MPS percent : $MPS_PERCENT"
echo "MPS pipe dir: $MPS_PIPE_DIRECTORY"
echo "MPS log dir : $MPS_LOG_DIRECTORY"
echo

shopt -s nullglob

SYSTEMD_RUN=(systemd-run)
SYSTEMCTL=(systemctl)
JOURNALCTL=(journalctl)
# system 模式下需要 root 创建 transient unit
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SYSTEMD_RUN=(sudo "${SYSTEMD_RUN[@]}")
  SYSTEMCTL=(sudo "${SYSTEMCTL[@]}")
  JOURNALCTL=(sudo "${JOURNALCTL[@]}")
fi

# 运行完成（包括失败）后自动卸载 transient unit，避免下次 --unit 同名冲突
# --no-block: 不阻塞当前 shell，允许循环继续启动下一个 client
SYSTEMD_RUN+=(--collect --scope --no-block)

ensure_mps_daemon() {
  if [[ "$MPS_ENABLE" != "1" ]]; then
    echo "MPS disabled by MPS_ENABLE=$MPS_ENABLE"
    return 0
  fi

  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "ERROR: nvidia-cuda-mps-control not found in PATH."
    echo "Install CUDA MPS tools or set MPS_ENABLE=0 to skip MPS."
    exit 1
  fi

  mkdir -p "$MPS_PIPE_DIRECTORY" "$MPS_LOG_DIRECTORY"

  if pgrep -u "$RUN_AS_USER" -f nvidia-cuda-mps-control >/dev/null 2>&1 || \
     pgrep -u "$RUN_AS_USER" -f nvidia-cuda-mps-server >/dev/null 2>&1; then
    echo "MPS daemon already running for user $RUN_AS_USER"
    return 0
  fi

  echo "Starting MPS daemon for user $RUN_AS_USER ..."
  if [[ "$(id -un)" == "$RUN_AS_USER" ]]; then
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY" \
    CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIRECTORY" \
      nvidia-cuda-mps-control -d
  else
    if ! command -v sudo >/dev/null 2>&1; then
      echo "ERROR: current user is $(id -un), target RUN_AS_USER is $RUN_AS_USER, but sudo is unavailable."
      exit 1
    fi
    sudo -u "$RUN_AS_USER" \
      CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY" \
      CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIRECTORY" \
      nvidia-cuda-mps-control -d
  fi

  sleep 0.2
  if pgrep -u "$RUN_AS_USER" -f nvidia-cuda-mps-control >/dev/null 2>&1 || \
     pgrep -u "$RUN_AS_USER" -f nvidia-cuda-mps-server >/dev/null 2>&1; then
    echo "MPS daemon started."
  else
    echo "ERROR: failed to start MPS daemon for user $RUN_AS_USER"
    exit 1
  fi
}

cleanup_unit() {
  local unit="$1"

  # If the unit exists from a previous run, stop/reset it first to avoid:
  # "Unit ... was already loaded or has a fragment file."
  if "${SYSTEMCTL[@]}" list-units --type=scope --all --no-legend "$unit" 2>/dev/null | awk '{print $1}' | grep -qx "$unit"; then
    echo "  Found existing unit: $unit; stopping/resetting it first..."
    "${SYSTEMCTL[@]}" stop "$unit" >/dev/null 2>&1 || true
    "${SYSTEMCTL[@]}" kill "$unit" >/dev/null 2>&1 || true
    "${SYSTEMCTL[@]}" reset-failed "$unit" >/dev/null 2>&1 || true
    # Give systemd a moment to GC collected transient units.
    for _ in {1..20}; do
      if ! "${SYSTEMCTL[@]}" list-units --type=scope --all --no-legend "$unit" 2>/dev/null | awk '{print $1}' | grep -qx "$unit"; then
        break
      fi
      sleep 0.1
    done
  fi
}

ensure_mps_daemon

count=0
launch_pids=()
declare -A CLIENT_CPU
declare -A CLIENT_AFFINITY
if [[ -f "$CPU_MAP_CSV" ]]; then
  cpu_map_args=(--cpu-map "$CPU_MAP_CSV")
  if [[ "$CPU_MAP_ONLY" == "1" && "$ENABLE_CPU_AFFINITY" == "1" ]]; then
    cpu_map_args+=(--require-distinct-affinity)
  fi
  cpu_rows=$("${PY_CMD[@]}" "$PROJECT_DIR/warmup_launch.py" "${cpu_map_args[@]}")
  while IFS=$'\t' read -r cid new_cpu affinity; do
    [[ -z "$cid" ]] && continue
    CLIENT_CPU["$cid"]="$new_cpu"
    if [[ "$ENABLE_CPU_AFFINITY" == "1" ]]; then
      CLIENT_AFFINITY["$cid"]="$affinity"
    fi
  done <<< "$cpu_rows"
fi
for f in "$DATA_DIR_ABS"/client_*.npz; do
  base=$(basename "$f")        # client_00035.npz
  cid=${base#client_}          # 00035.npz
  cid=${cid%.npz}              # 00035
  cid=$((10#$cid))             # 去掉前导 0，转成整数
  if [[ "$CPU_MAP_ONLY" == "1" && -z "${CLIENT_CPU[$cid]+set}" ]]; then
    continue
  fi

  unit="fl_client_${cid}"
  unit_name="${unit}.scope"
  log_file="$CLIENT_LOG_DIR/process_${cid}.log"

  echo "Starting client for $base (cid=$cid)"
  echo "  unit : $unit_name"
  echo "  log  : $log_file"

  cleanup_unit "$unit_name"

  client_cmd=(
    "${PY_CMD[@]}" "$PROJECT_DIR/run_client.py"
    --cid "$cid"
    --server "$SERVER"
    --data-dir "$DATA_DIR_ABS"
    --dataset "$DATASET"
    --model "$MODEL"
    --lr "$LR"
    --num-classes "$NUM_CLASSES"
    --batch-size "$BATCH_SIZE"
    --uplink-num-bits 0
    --log-file "$CLIENT_LOG_DIR/training_${cid}.log"
    --training-timing-jsonl "$CLIENT_LOG_DIR/client_${cid}.jsonl"
    --measurement-run-id "${MEASUREMENT_RUN_ID:-$(basename "$CLIENT_LOG_DIR")}"
    --cpu-fraction "${CLIENT_CPU[$cid]:-1.00}"
  )
  if [[ -n "$LOCAL_EPOCHS" ]]; then
    client_cmd+=(--epochs "$LOCAL_EPOCHS")
  fi
  if [[ "$PACER_ENABLE" == "1" ]]; then
    client_cmd+=(--pacer)
    if [[ -n "$PACER_LOG_DIR" ]]; then
      mkdir -p "$PACER_LOG_DIR"
      client_cmd+=(--pacer-log "$PACER_LOG_DIR/pacer_client_${cid}.jsonl")
    fi
    if [[ -n "$PACER_RESOURCE_STATE_DIR" ]]; then
      client_cmd+=(--pacer-resource-state "$PACER_RESOURCE_STATE_DIR/client_${cid}.json")
    fi
  fi
  affinity="-"
  if [[ "$ENABLE_CPU_AFFINITY" == "1" ]]; then
    affinity="${CLIENT_AFFINITY[$cid]:--}"
    if [[ "$affinity" == "-" && "$BIND_CLIENT_TO_CPU" == "1" ]]; then
      cpu_count=$(nproc)
      affinity=$((cid + CPU_AFFINITY_OFFSET))
      if [[ "$CPU_AFFINITY_WRAP" == "1" ]]; then
        affinity=$((affinity % cpu_count))
      elif (( affinity < 0 || affinity >= cpu_count )); then
        echo "ERROR: client $cid maps to CPU $affinity, but available CPUs are 0-$((cpu_count - 1))"
        exit 1
      fi
    fi
  fi
  if [[ "$affinity" != "-" ]]; then
    client_cmd+=(--cpu-affinity "$affinity")
    client_cmd=(taskset -c "$affinity" "${client_cmd[@]}")
  fi
  printf '%s,%s,"%s"\n' "$cid" "${CLIENT_CPU[$cid]:-1.00}" "$affinity" >> "$CLIENT_LOG_DIR/cpu_config.csv"

  cmd_str="$(printf '%q ' "${client_cmd[@]}")"
  log_q="$(printf '%q' "$log_file")"

  cpu_frac="${CLIENT_CPU[$cid]:-1.00}"
  cpu_quota=$(awk -v c="$cpu_frac" 'BEGIN{gsub(/[ \t\r]/,"",c); if(c=="") c=1.00; printf "%.2f%%", c*100}')
  "${SYSTEMD_RUN[@]}" \
    -p CPUAccounting=yes -p CPUQuota="$cpu_quota" -p CPUQuotaPeriodSec=20ms \
    --uid="$RUN_AS_USER" \
    --working-directory="$PROJECT_DIR" \
    "${ENV_VARS[@]}" \
    --unit="$unit" \
    /bin/bash -lc "${cmd_str} >> ${log_q} 2>&1" &
  launch_pids+=("$!")

  # 如果 unit 立即失败，直接给出原因（常见：用户/环境/模块路径）
  if ! "${SYSTEMCTL[@]}" is-active --quiet "$unit_name"; then
    echo "  WARN: $unit_name is not active; check:"
    echo "    ${SYSTEMCTL[*]} status $unit_name -n 50 --no-pager"
    echo "    ${JOURNALCTL[*]} -u $unit_name -n 100 --no-pager"
  fi

  count=$((count + 1))
  if [[ "$MAX_CLIENTS" -gt 0 && "$count" -ge "$MAX_CLIENTS" ]]; then
    echo "Reached MAX_CLIENTS=$MAX_CLIENTS, stop launching more."
    break
  fi
done

launch_failed=0
for pid in "${launch_pids[@]}"; do
  wait "$pid" || launch_failed=1
done

if [[ $count -eq 0 ]]; then
  echo "No client_*.npz found in $DATA_DIR"
else
  echo "Launched $count clients."
fi
exit "$launch_failed"
