#!/usr/bin/env bash
# push_fedavgm_scp.sh  —— 仅用 scp/ssh，无 sshpass
set -Eeuo pipefail

# ======= 配置 =======
PROJECT_DIR="/home/xiaoyan/wholeflower"   # ← 改成你的项目根目录
HOSTS_FILE="./hosts.txt"              # 每行一个：ip 或 user@ip（顺序即分配顺序）
DEFAULT_USER="pi"                     # hosts.txt 若无 user@，用它
REMOTE_SUBDIR="femnist"               # 远端放到 ~/femnist
SSH_OPTS="-o StrictHostKeyChecking=accept-new"  # 老机器不支持时可换成：-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
# 如需自动输入密码，可在此填写（明文存储存在安全风险，建议优先使用密钥认证）
SSH_PASSWORD="abacus"
# 需要额外同步的文件（使用项目根目录的相对路径，保持目录结构）
PATCH_FILES=(
)
# ====================

ZIP="$PROJECT_DIR/shakespeare.zip"
CLIENT_DIR="$PROJECT_DIR/femnist/data_partitions"
SYNC_ZIP=false              # 设为 true 则继续同步 ZIP
SYNC_CLIENT_DATA=true      # 设为 true 则继续同步 client_*.npz

# 如果需要同步 ZIP / client 数据，则检查文件是否存在
if $SYNC_ZIP; then
  [[ -f "$ZIP" ]] || { echo "找不到 $ZIP"; exit 1; }
fi
if $SYNC_CLIENT_DATA; then
  [[ -d "$CLIENT_DIR" ]] || { echo "找不到目录 $CLIENT_DIR"; exit 1; }
fi

# 检查 PATCH 文件是否存在
for file in "${PATCH_FILES[@]}"; do
  if [[ ! -f "$PROJECT_DIR/$file" ]]; then
    echo "缺少补丁文件: $PROJECT_DIR/$file"
    exit 1
  fi
done

# 告知将同步的补丁文件
echo "[*] 将同步以下补丁文件到客户端："
for rel in "${PATCH_FILES[@]}"; do
  echo "    - $rel"
done

read -ra SSH_OPTS_ARR <<< "$SSH_OPTS"
if [[ -n "$SSH_PASSWORD" ]]; then
  command -v sshpass >/dev/null 2>&1 || { echo "已设置 SSH_PASSWORD 但未找到 sshpass，请先安装 sshpass 或清空密码配置"; exit 1; }
  SSH_BIN=(sshpass -p "$SSH_PASSWORD" ssh "${SSH_OPTS_ARR[@]}")
  SCP_BIN=(sshpass -p "$SSH_PASSWORD" scp "${SSH_OPTS_ARR[@]}" -q)
else
  SSH_BIN=(ssh "${SSH_OPTS_ARR[@]}")
  SCP_BIN=(scp "${SSH_OPTS_ARR[@]}" -q)
fi

# 读取主机列表（去掉空行/注释）
mapfile -t HOSTS < <(grep -v '^\s*#' "$HOSTS_FILE" | sed '/^\s*$/d')
N=${#HOSTS[@]}
(( N > 0 )) || { echo "hosts.txt 为空"; exit 1; }
echo "[*] 主机数: $N"

parse_user_host() {  # 输入：一行 host；输出：user host
  local entry="$1"
  if [[ "$entry" == *"@"* ]]; then
    printf '%s %s\n' "${entry%@*}" "${entry#*@}"
  else
    printf '%s %s\n' "$DEFAULT_USER" "$entry"
  fi
}

for i in "${!HOSTS[@]}"; do
  read -r user host < <(parse_user_host "${HOSTS[$i]}")
  echo "==> [$((i+1))/$N] ${user}@${host}"

  # 远端目标目录（~ 在远端展开）
  remote_dir="~/${REMOTE_SUBDIR}"

  # 先创建远端目录
  "${SSH_BIN[@]}" "${user}@${host}" "mkdir -p ${remote_dir}"

  # 1) 传 fedavgm.zip（可选）
  if $SYNC_ZIP; then
    "${SCP_BIN[@]}" "$ZIP" "${user}@${host}:${remote_dir}/" \
      && echo "   [+] fedavgm.zip 已传"
  else
    echo "   [i] 跳过 fedavgm.zip"
  fi

  # 2) 传对应的 client_0000x.npz（只给前20台，可选）
  if $SYNC_CLIENT_DATA; then
    if (( i < 20 )); then
      idx=$(printf "%05d" "$i")                     # 00000 .. 00019
      client_npz="$CLIENT_DIR/client_${idx}.npz"
      if [[ -f "$client_npz" ]]; then
        "${SCP_BIN[@]}" "$client_npz" "${user}@${host}:${remote_dir}/" \
          && echo "   [+] $(basename "$client_npz") 已传"
      else
        echo "   [!] 缺少文件：$client_npz（已跳过）"
      fi
    else
      echo "   [i] 超出前20台，只传 zip"
    fi
  else
    echo "   [i] 跳过 client_*.npz"
  fi

  # 3) 传补丁文件
  for rel in "${PATCH_FILES[@]}"; do
    local_path="$PROJECT_DIR/$rel"
    rel_stripped="${rel#*/}"
    remote_path="$remote_dir/$rel_stripped"
    remote_parent=$(dirname "$remote_path")
    "${SSH_BIN[@]}" "${user}@${host}" "mkdir -p ${remote_parent}"
    "${SCP_BIN[@]}" "$local_path" "${user}@${host}:${remote_path}" \
      && echo "   [+] $rel 已同步"
  done
done

echo "全部完成。"
