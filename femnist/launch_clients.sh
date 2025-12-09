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

# 当前项目目录（你在 femnist 代码根目录下执行脚本就行）
PROJECT_DIR=$(pwd)
# 把 data_dir 变成绝对路径，避免工作目录不同导致找不到
DATA_DIR_ABS=$(readlink -f "$DATA_DIR")

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
echo "Project dir : $PROJECT_DIR"
echo

shopt -s nullglob

count=0
for f in "$DATA_DIR"/client_*.npz; do
  base=$(basename "$f")        # client_00035.npz
  cid=${base#client_}          # 00035.npz
  cid=${cid%.npz}              # 00035
  cid=$((10#$cid))             # 去掉前导 0，转成整数

  echo "Starting client for $base (cid=$cid)"

  sudo systemd-run \
    -p CPUQuota=18.6% -p CPUQuotaPeriodSec=200ms \
    --uid=ubuntu \
    --working-directory="$PROJECT_DIR" \
    "${ENV_VARS[@]}" \
    --unit="fl_client_${cid}" \
    "$PY" -m run_client \
      --cid "$cid" \
      --server "$SERVER" \
      --data-dir "$DATA_DIR_ABS" \
      --model mobilenet_v2_100 \
      --num-classes 62 \
      --uplink-num-bits 0

  count=$((count + 1))
  if [[ "$MAX_CLIENTS" -gt 0 && "$count" -ge "$MAX_CLIENTS" ]]; then
    echo "Reached MAX_CLIENTS=$MAX_CLIENTS, stop launching more."
    break
  fi
done

if [[ $count -eq 0 ]]; then
  echo "No client_*.npz found in $DATA_DIR"
else
  echo "Launched $count clients."
fi
