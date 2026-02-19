#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE=<your-image> ./launch_clients_k8s.sh [DATA_DIR] [SERVER_ADDR] [MAX_CLIENTS] [NAMESPACE]

DATA_DIR=${1:-data_partitions_speech_commands}
SERVER=${2:-"127.0.0.1:8081"}
MAX_CLIENTS=${3:-3}
NAMESPACE=${4:-client-a}

IMAGE=${IMAGE:-}
IMAGE_PULL_POLICY=${IMAGE_PULL_POLICY:-IfNotPresent}
DATASET=${DATASET:-speech_commands}
MODEL=${MODEL:-resnet34}
LR=${LR:-0.01}
BATCH_SIZE=${BATCH_SIZE:-20}
UPLINK_NUM_BITS=${UPLINK_NUM_BITS:-0}
CLIENT_ENTRY=${CLIENT_ENTRY:-run_client_torch.py}

CPU_REQUEST=${CPU_REQUEST:-500m}
CPU_LIMIT=${CPU_LIMIT:-2}
MEM_REQUEST=${MEM_REQUEST:-2Gi}
MEM_LIMIT=${MEM_LIMIT:-8Gi}

GPU_LIMIT=${GPU_LIMIT:-1}
GPU_MEM=${GPU_MEM:-10000}
GPU_CORES=${GPU_CORES:-10}
RUN_AS_UID=${RUN_AS_UID:-1001}
RUN_AS_GID=${RUN_AS_GID:-1001}

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$DATA_DIR" = /* ]]; then
  DATA_DIR_ABS="$(readlink -f "$DATA_DIR")"
else
  DATA_DIR_ABS="$(readlink -f "$PROJECT_DIR/$DATA_DIR")"
fi

if [[ -z "$IMAGE" ]]; then
  echo "ERROR: IMAGE is required."
  exit 1
fi

kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

shopt -s nullglob
count=0

for f in "$DATA_DIR_ABS"/client_*.npz; do
  base="$(basename "$f")"
  cid="${base#client_}"
  cid="${cid%.npz}"
  cid="$((10#$cid))"

  pod="fl-client-${cid}"
  echo "Launching $pod from $base"

  # 强制删除旧 Pod，确保新配置生效
  kubectl -n "$NAMESPACE" delete pod "$pod" --ignore-not-found=true --force --grace-period=0 >/dev/null 2>&1 || true

  # 生成 Pod YAML
  # 注意：下面的 $REAL_CUDART 等变量前都加了 \，这是为了防止在宿主机报错
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
  namespace: ${NAMESPACE}
  labels:
    app: fl-client
    client-id: "${cid}"
  annotations:
    hami.io/gpu-memory: "${GPU_MEM}"
    hami.io/gpu-core: "${GPU_CORES}"
spec:
  restartPolicy: Never
  runtimeClassName: nvidia
  schedulerName: hami-scheduler
  nodeSelector:
    gpu: "on"
  securityContext:
    runAsUser: ${RUN_AS_UID}
    runAsGroup: ${RUN_AS_GID}
    fsGroup: ${RUN_AS_GID}
  containers:
  - name: client
    image: ${IMAGE}
    imagePullPolicy: ${IMAGE_PULL_POLICY}
    # [核心修复] 使用 Shell 包装器来控制加载顺序
    command: ["/bin/bash", "-c"]
    args:
      - |
        echo "--- Setting up vGPU environment ---"
        
        # 1. 查找真正的 CUDA 库 (宿主机不要解析这些变量，所以加了反斜杠)
        # 我们优先找 libcudart.so.1*，通常在 /usr/local/cuda 或 /usr/lib
        REAL_CUDART=\$(find /usr -name "libcudart.so.1*" 2>/dev/null | head -n 1)
        
        # 2. 查找 HAMI 拦截库
        HAMI_LIB=\$(find /usr/local/vgpu -name "libvgpu.so*" 2>/dev/null | head -n 1)
        
        if [ -z "\$REAL_CUDART" ]; then
           echo "WARNING: libcudart.so not found! Trying default path..."
           REAL_CUDART="/usr/local/cuda/lib64/libcudart.so.11.0"
        else
           echo "Found CUDART: \$REAL_CUDART"
        fi

        if [ -z "\$HAMI_LIB" ]; then
           echo "ERROR: libvgpu.so not found in /usr/local/vgpu! Check mounts."
        else
           echo "Found HAMI LIB: \$HAMI_LIB"
        fi

        # 3. [关键] 设置加载顺序：先 CUDA，后 HAMI
        # 这能避免 TF 初始化时的死锁
        export LD_PRELOAD="\$REAL_CUDART:\$HAMI_LIB"
        echo "Final LD_PRELOAD: \$LD_PRELOAD"
        
        echo "--- Starting Client Python Script ---"
        
        # 启动 Python (这里可以使用宿主机的变量，如 ${cid})
        python /workspace/${CLIENT_ENTRY} \
        --cid ${cid} \
        --server ${SERVER} \
        --data-dir /data \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --lr ${LR} \
        --batch-size ${BATCH_SIZE} \
        --uplink-num-bits ${UPLINK_NUM_BITS}
    env:
      - name: OMP_NUM_THREADS
        value: "1"
      # [K3s 修复核心 1] 让 Pod 知道自己的 UID
      - name: POD_UID
        valueFrom:
          fieldRef:
            fieldPath: metadata.uid
            
      # [K3s 修复核心 2] 强制 libvgpu 使用 Pod UID 作为握手文件名
      # 这样 Monitor 才能在 K8s API 里找到对应的 Pod
      - name: VGPU_CONTAINER_ID
        valueFrom:
          fieldRef:
            fieldPath: metadata.uid
            
      # [K3s 修复核心 3] 补充元数据，防止服务端查询失败
      - name: POD_NAME
        valueFrom:
          fieldRef:
            fieldPath: metadata.name
      - name: POD_NAMESPACE
        valueFrom:
          fieldRef:
            fieldPath: metadata.namespace
    resources:
      limits:
        nvidia.com/gpu: "${GPU_LIMIT}"
        nvidia.com/gpumem: "${GPU_MEM}"
        nvidia.com/gpucores: "${GPU_CORES}"
        cpu: ${CPU_LIMIT}
        memory: ${MEM_LIMIT}
      requests:
        cpu: ${CPU_REQUEST}
        memory: ${MEM_REQUEST}
    volumeMounts:
      - name: project
        mountPath: /workspace
      - name: data
        mountPath: /data
      # 挂载 HAMI 通信目录
      - name: hami-communication-dir
        mountPath: /usr/local/vgpu
        readOnly: false
  volumes:
    - name: project
      hostPath:
        path: ${PROJECT_DIR}
        type: Directory
    - name: data
      hostPath:
        path: ${DATA_DIR_ABS}
        type: Directory
    - name: hami-communication-dir
      hostPath:
        path: /usr/local/vgpu
        type: Directory
EOF

  count=$((count + 1))
  if [[ "$MAX_CLIENTS" -gt 0 && "$count" -ge "$MAX_CLIENTS" ]]; then
    break
  fi
done
