#!/usr/bin/env bash
set -euo pipefail

# 使用方式：
#   PY=/path/to/python ./launch_clients.sh [DATA_DIR] [SERVER_ADDR] [MAX_CLIENTS] [CPU_QUOTA]
#
# 例子：
#   PY=$(which python3) ./launch_clients.sh data_partitions 172.31.17.220:8081 42 60
#
# 可选环境变量：
#   PLAN_CSV=/path/to/slo_plan.csv PLAN_MODE=fedpacer ./launch_clients.sh ...
#   UNIT_PREFIX=fl_client_warmup_40_ SYSTEMD_UID=$(id -un) ./launch_clients.sh ...
#   DATASET=cifar10 MODEL=resnet18 LR=0.01 LOCAL_STEPS=20 BATCH_SIZE=128 ./launch_clients.sh ...
#   DATASET=ixi DATA_ROOT=/home/xiaoyan/wholeflower/data/fed_ixi LR=0.001 BATCH_SIZE=2 ./launch_clients.sh
#   REPLACE_EXISTING_UNITS=0 ./launch_clients.sh ...   # 不清理同名旧 unit

# 当前脚本所在目录，也就是 femnist 代码根目录。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATASET=${DATASET:-femnist}
if [[ $# -ge 1 ]]; then
  if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
    DATA_ROOT=$1
    DATA_DIR=${DATA_DIR:-}
  else
    DATA_DIR=$1
  fi
elif [[ "$DATASET" == "cifar10" || "$DATASET" == "cifar" || "$DATASET" == "cifar-10" ]]; then
  DATA_DIR=data_partitions_cifar10
elif [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  DATA_ROOT=${DATA_ROOT:-"$SCRIPT_DIR/../data/fed_ixi"}
  DATA_DIR=${DATA_DIR:-}
else
  DATA_DIR=data_partitions
fi
SERVER=${2:-"127.0.0.1:8081"}
MAX_CLIENTS=${3:-0}   # 0 = 启动目录里所有 client_*.npz
CPU_QUOTA=${4:-${CPU_QUOTA:-80}}

if [[ -z "${PY:-}" && -x "$SCRIPT_DIR/../venv/bin/python" ]]; then
  PY="$SCRIPT_DIR/../venv/bin/python"
else
  PY=${PY:-python3}
fi
SYSTEMD_UID=${SYSTEMD_UID:-$(id -un)}
UNIT_PREFIX=${UNIT_PREFIX:-fl_client_}
PLAN_CSV=${PLAN_CSV:-}
PLAN_MODE=${PLAN_MODE:-}
REPLACE_EXISTING_UNITS=${REPLACE_EXISTING_UNITS:-1}
LOCAL_STEPS=${LOCAL_STEPS:-20}
if [[ "$DATASET" == "cifar10" || "$DATASET" == "cifar" || "$DATASET" == "cifar-10" ]]; then
  MODEL=${MODEL:-resnet18}
  LR=${LR:-0.01}
  BATCH_SIZE=${BATCH_SIZE:-128}
  NUM_CLASSES=${NUM_CLASSES:-10}
elif [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  # run_client.py selects UNet3D(in_channels=1, out_channels=2, base=8)
  # automatically for dataset=ixi. The --model flag is intentionally not
  # passed for IXI because argparse choices do not include "unet".
  LR=${LR:-0.001}
  BATCH_SIZE=${BATCH_SIZE:-2}
  NUM_CLASSES=${NUM_CLASSES:-2}
  IXI_CLIENT_IDS=${IXI_CLIENT_IDS:-"0 1 2"}  # 0=Guys, 1=HH, 2=IOP
else
  MODEL=${MODEL:-cnn}
  LR=${LR:-0.003}
  BATCH_SIZE=${BATCH_SIZE:-64}
  NUM_CLASSES=${NUM_CLASSES:-62}
fi

PROJECT_DIR=${PROJECT_DIR:-$SCRIPT_DIR}

if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  if [[ "$DATA_ROOT" = /* ]]; then
    DATA_ROOT_ABS=$(readlink -f "$DATA_ROOT")
  else
    DATA_ROOT_ABS=$(readlink -f "$PROJECT_DIR/$DATA_ROOT")
  fi
else
  # 把 data_dir 变成绝对路径，避免工作目录不同导致找不到
  if [[ "$DATA_DIR" = /* ]]; then
    DATA_DIR_ABS=$(readlink -f "$DATA_DIR")
  else
    DATA_DIR_ABS=$(readlink -f "$PROJECT_DIR/$DATA_DIR")
  fi
fi

ENV_VARS=(
  --setenv=OMP_NUM_THREADS=1
  --setenv=OPENBLAS_NUM_THREADS=1
  --setenv=MKL_NUM_THREADS=1
  --setenv=NUMEXPR_NUM_THREADS=1
  --setenv=TF_NUM_INTRAOP_THREADS=1
  --setenv=TF_NUM_INTEROP_THREADS=1
)

if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  echo "Data root   : $DATA_ROOT_ABS"
  echo "IXI clients : $IXI_CLIENT_IDS"
else
  echo "Data dir    : $DATA_DIR_ABS"
fi
echo "Server      : $SERVER"
echo "Max clients : ${MAX_CLIENTS:-all}"
echo "CPU quota   : ${CPU_QUOTA}%"
echo "Unit prefix : $UNIT_PREFIX"
echo "Replace old : $REPLACE_EXISTING_UNITS"
if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  echo "Model       : UNet3D(in_channels=1,out_channels=2,base=8)"
else
  echo "Model       : $MODEL"
fi
echo "Dataset     : $DATASET"
echo "Num classes : $NUM_CLASSES"
echo "LR          : $LR"
echo "Local steps : $LOCAL_STEPS"
echo "Batch size  : $BATCH_SIZE"
if [[ -n "$PLAN_CSV" ]]; then
  if [[ ! -f "$PLAN_CSV" ]]; then
    echo "Plan CSV not found: $PLAN_CSV" >&2
    exit 1
  fi
  echo "Plan CSV    : $PLAN_CSV"
  echo "Plan mode   : ${PLAN_MODE:-all}"
fi
echo "Python      : $PY"
echo "Project dir : $PROJECT_DIR"
echo

quota_for_client() {
  local cid="$1"
  if [[ -z "$PLAN_CSV" ]]; then
    normalize_cpu_quota "$CPU_QUOTA"
    return
  fi
  python3 - "$PLAN_CSV" "$PLAN_MODE" "$cid" "$CPU_QUOTA" <<'PY'
import csv
import math
import sys

path, mode, cid, default = sys.argv[1:5]
matched_mode = False
value = None
with open(path, newline="", encoding="utf-8") as f:
    rows = csv.DictReader(f)
    for row in rows:
        if mode and row.get("mode") != mode:
            continue
        matched_mode = True
        if str(row.get("client_id", "")).strip() == str(int(cid)):
            value = row.get("cpu_quota", default)
            break
if value is None:
    if mode and not matched_mode:
        print(f"Plan CSV has no rows for PLAN_MODE={mode!r}", file=sys.stderr)
    else:
        print(f"Plan CSV has no cpu_quota for client_id={int(cid)} mode={mode or 'all'}", file=sys.stderr)
    raise SystemExit(2)
quota = float(value)
if not math.isfinite(quota) or quota <= 0.0:
    raise SystemExit(f"invalid CPUQuota value: {value}")
print(int(math.ceil(quota)))
PY
}

normalize_cpu_quota() {
  python3 - "$@" <<'PY'
import math
import sys

raw = sys.argv[1].strip() if len(sys.argv) > 1 else sys.stdin.read().strip()
if not raw:
    raise SystemExit(2)
value = float(raw)
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(f"invalid CPUQuota value: {raw}")
print(int(math.ceil(value)))
PY
}

cleanup_unit() {
  local unit="$1"
  if [[ "$REPLACE_EXISTING_UNITS" != "1" ]]; then
    return
  fi

  # systemd-run refuses to reuse a loaded transient unit name until it is
  # stopped and any failed state is reset.
  sudo systemctl stop "$unit.service" >/dev/null 2>&1 || true
  sudo systemctl reset-failed "$unit.service" >/dev/null 2>&1 || true
}

shopt -s nullglob

count=0
if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
  for cid in $IXI_CLIENT_IDS; do
    client_cpu_quota=$(quota_for_client "$cid")
    unit_name="${UNIT_PREFIX}${cid}"

    cleanup_unit "$unit_name"

    echo "Starting IXI client cid=$cid (unit=${unit_name}.service, CPUQuota=${client_cpu_quota}%)"

    sudo systemd-run \
      -p CPUQuota="${client_cpu_quota}%" -p CPUQuotaPeriodSec=200ms \
      --uid="$SYSTEMD_UID" \
      --working-directory="$PROJECT_DIR" \
      "${ENV_VARS[@]}" \
      --unit="$unit_name" \
      "$PY" -m run_client \
        --cid "$cid" \
        --dataset "$DATASET" \
        --server "$SERVER" \
        --data-root "$DATA_ROOT_ABS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "${NUM_WORKERS:-0}" \
        --num-classes "$NUM_CLASSES" \
        --uplink-num-bits "${UPLINK_NUM_BITS:-0}"

    count=$((count + 1))
    if [[ "$MAX_CLIENTS" -gt 0 && "$count" -ge "$MAX_CLIENTS" ]]; then
      echo "Reached MAX_CLIENTS=$MAX_CLIENTS, stop launching more."
      break
    fi
  done
else
for f in "$DATA_DIR"/client_*.npz; do
  base=$(basename "$f")        # client_00035.npz
  cid=${base#client_}          # 00035.npz
  cid=${cid%.npz}              # 00035
  cid=$((10#$cid))             # 去掉前导 0，转成整数
  client_cpu_quota=$(quota_for_client "$cid")
  unit_name="${UNIT_PREFIX}${cid}"

  cleanup_unit "$unit_name"

  echo "Starting client for $base (cid=$cid, unit=${unit_name}.service, CPUQuota=${client_cpu_quota}%)"

  sudo systemd-run \
    -p CPUQuota="${client_cpu_quota}%" -p CPUQuotaPeriodSec=200ms \
    --uid="$SYSTEMD_UID" \
    --working-directory="$PROJECT_DIR" \
    "${ENV_VARS[@]}" \
    --unit="$unit_name" \
    "$PY" -m run_client \
      --cid "$cid" \
      --dataset "$DATASET" \
      --server "$SERVER" \
      --data-dir "$DATA_DIR_ABS" \
      --model "$MODEL" \
      --lr "$LR" \
      --local-steps "$LOCAL_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --num-classes "$NUM_CLASSES" \
      --uplink-num-bits 0

  count=$((count + 1))
  if [[ "$MAX_CLIENTS" -gt 0 && "$count" -ge "$MAX_CLIENTS" ]]; then
    echo "Reached MAX_CLIENTS=$MAX_CLIENTS, stop launching more."
    break
  fi
done
fi

if [[ $count -eq 0 ]]; then
  if [[ "$DATASET" == "ixi" || "$DATASET" == "fed-ixi" || "$DATASET" == "fed_ixi" || "$DATASET" == "fedixi" ]]; then
    echo "No IXI clients launched. Check IXI_CLIENT_IDS."
  else
    echo "No client_*.npz found in $DATA_DIR"
  fi
else
  echo "Launched $count clients."
fi
