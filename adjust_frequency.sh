#!/usr/bin/env bash
set -euo pipefail

# SmartPC-style frequency control:
# 1) Preferred path: userspace governor + scaling_setspeed.
# 2) Fallback path (e.g., intel_pstate active): performance + min=max.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo $0 <freq_mhz|freq_khz>"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: sudo $0 <freq_mhz|freq_khz>"
  echo "Example: sudo $0 3000    # 3000 MHz"
  echo "Example: sudo $0 3000000 # 3000000 kHz"
  exit 1
fi

input_freq="$1"
if [[ ! "$input_freq" =~ ^[0-9]+$ ]]; then
  echo "Frequency must be an integer."
  exit 1
fi

# Treat values below 10000 as MHz.
if (( input_freq < 10000 )); then
  target_khz=$((input_freq * 1000))
else
  target_khz=$input_freq
fi

policy_root="/sys/devices/system/cpu/cpufreq"
policies=("${policy_root}"/policy*)
if [[ ! -e "${policies[0]}" ]]; then
  echo "No cpufreq policy found under ${policy_root}."
  exit 1
fi

policy0="${policies[0]}"
available_governors="$(<"${policy0}/scaling_available_governors")"

pick_nearest_khz() {
  local policy="$1"
  local req="$2"
  local min max
  min="$(<"${policy}/cpuinfo_min_freq")"
  max="$(<"${policy}/cpuinfo_max_freq")"

  if (( req < min )); then
    req=$min
  elif (( req > max )); then
    req=$max
  fi

  # If the platform exposes discrete levels, pick the nearest one.
  if [[ -f "${policy}/scaling_available_frequencies" ]]; then
    local best=0 best_diff=999999999 f diff
    for f in $(<"${policy}/scaling_available_frequencies"); do
      diff=$((f > req ? f - req : req - f))
      if (( diff < best_diff )); then
        best="$f"
        best_diff="$diff"
      fi
    done
    echo "$best"
    return
  fi

  echo "$req"
}

if grep -qw "userspace" <<<"${available_governors}"; then
  mode="userspace+setspeed"
  for p in "${policies[@]}"; do
    freq="$(pick_nearest_khz "$p" "$target_khz")"
    echo userspace >"${p}/scaling_governor"
    echo "$freq" >"${p}/scaling_setspeed"
    # Keep the range fixed to avoid governor interference.
    echo "$freq" >"${p}/scaling_min_freq"
    echo "$freq" >"${p}/scaling_max_freq"
  done
else
  mode="performance+min=max (userspace not supported)"
  # For intel_pstate, keep turbo enabled when requesting frequencies above base.
  if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]] && [[ -f "${policy0}/base_frequency" ]]; then
    base_khz="$(<"${policy0}/base_frequency")"
    if (( target_khz > base_khz )); then
      echo 0 >/sys/devices/system/cpu/intel_pstate/no_turbo
    fi
  fi

  for p in "${policies[@]}"; do
    freq="$(pick_nearest_khz "$p" "$target_khz")"
    echo performance >"${p}/scaling_governor"
    echo "$freq" >"${p}/scaling_min_freq"
    echo "$freq" >"${p}/scaling_max_freq"
  done
fi

echo "Applied mode: ${mode}"
for p in "${policies[@]}"; do
  printf "%s gov=%s min=%s max=%s cur=%s\n" \
    "$(basename "$p")" \
    "$(<"${p}/scaling_governor")" \
    "$(<"${p}/scaling_min_freq")" \
    "$(<"${p}/scaling_max_freq")" \
    "$(<"${p}/scaling_cur_freq")"
done
