#!/usr/bin/env bash
set -euo pipefail

# 使用方式：
#   PY=/path/to/python ./launch_clients.sh [DATA_DIR] [SERVER_ADDR] [MAX_CLIENTS]
#
# 例子：
#   PY=$(which python3) ./launch_clients.sh data_partitions 172.31.17.220:8081 42

DATA_DIR=${1:-data_partitions}
SERVER=${2:-"127.0.0.1:8081"}
MAX_CLIENTS=${3:-0}   # 0 = 启动目录里所有 client_*.npz

PY=${PY:-python3}

# 运行哪个用户（默认当前用户）；之前写死 ubuntu 会导致 217/USER，客户端根本没起来
RUN_AS_USER=${RUN_AS_USER:-"$(id -un)"}

# 强制使用 systemd system + scope 模式（与旧的 --scope 行为一致）
SYSTEMD_MODE="system"

# femnist 项目目录（脚本所在目录），避免从别的路径执行导致 working-directory/data-dir 错乱
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Optional: CSV mapping client_id -> new_cpu (fraction 0-1). If set, use per-client CPUQuota.
CPU_MAP_CSV=${CPU_MAP_CSV:-"$PROJECT_DIR/../logs/client_num_examples.csv"}

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
)

echo "Data dir    : $DATA_DIR_ABS"
echo "Server      : $SERVER"
echo "Max clients : ${MAX_CLIENTS:-all}"
echo "Python      : $PY"
echo "Run as user : $RUN_AS_USER"
echo "Project dir : $PROJECT_DIR"
echo "CPU map CSV : $CPU_MAP_CSV"
echo "CPU affinity: disabled"
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

count=0
launch_pids=()
declare -A CLIENT_CPU
if [[ -f "$CPU_MAP_CSV" ]]; then
  while IFS=, read -r cid _ _ new_cpu; do
    [[ "$cid" == "client_id" || -z "$cid" ]] && continue
    cid=$(echo "$cid" | tr -d ' \t\r')
    new_cpu=$(echo "$new_cpu" | tr -d ' \t\r')
    CLIENT_CPU["$cid"]="$new_cpu"
  done < "$CPU_MAP_CSV"
fi
for f in "$DATA_DIR_ABS"/client_*.npz; do
  base=$(basename "$f")        # client_00035.npz
  cid=${base#client_}          # 00035.npz
  cid=${cid%.npz}              # 00035
  cid=$((10#$cid))             # 去掉前导 0，转成整数

  unit="fl_client_${cid}"
  unit_name="${unit}.scope"
  log_file="$DATA_DIR_ABS/${base%.npz}.log"

  echo "Starting client for $base (cid=$cid)"
  echo "  unit : $unit_name"
  echo "  log  : $log_file"

  cleanup_unit "$unit_name"

  client_cmd=(
    "$PY" "$PROJECT_DIR/run_client.py"
    --cid "$cid"
    --server "$SERVER"
    --data-dir "$DATA_DIR_ABS"
    --model cnn
    --num-classes 10
    --uplink-num-bits 0
  )

  cmd_str="$(printf '%q ' "${client_cmd[@]}")"
  log_q="$(printf '%q' "$log_file")"

  cpu_frac="${CLIENT_CPU[$cid]:-1.00}"
  cpu_quota=$(awk -v c="$cpu_frac" 'BEGIN{gsub(/[ \t\r]/,"",c); if(c=="") c=1.00; printf "%.2f%%", c*100}')
  "${SYSTEMD_RUN[@]}" \
    -p CPUAccounting=yes -p CPUQuota="$cpu_quota" -p CPUQuotaPeriodSec=200ms \
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

for pid in "${launch_pids[@]}"; do
  wait "$pid" || true
done

if [[ $count -eq 0 ]]; then
  echo "No client_*.npz found in $DATA_DIR"
else
  echo "Launched $count clients."
fi
