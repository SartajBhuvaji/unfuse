#!/usr/bin/env bash
# Capture the exact machine a benchmark ran on.
#
# A throughput number without its environment is not a measurement, it is an
# anecdote. Two runs of the same model on the same GPU model can differ by more
# than the effect you are trying to report -- driver version, CUDA version,
# clock/power caps, and whether ECC is on all move the number. This prints the
# provenance block that belongs next to every table in a model card.
#
# Usage:  bash tools/gpu_env.sh            # human readable
#         bash tools/gpu_env.sh --json     # machine readable, for a results dataset
set -uo pipefail

json=0; [ "${1:-}" = "--json" ] && json=1

# Which interpreter's torch to report. The ambient python is almost never the
# one that served the benchmark: vLLM and SGLang each need their own venv, and
# a Lambda Stack box ships a system torch that neither of them uses. Recording
# the wrong torch next to a throughput table is a provenance error, so point
# this at the venv that actually ran the model.
#   PYTHON=~/venvs/vllm/bin/python bash tools/gpu_env.sh --json
PY_BIN="${PYTHON:-python3}"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
cuda_smi=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | awk '{print $3}')
nvcc_ver=$(nvcc --version 2>/dev/null | grep -oE "release [0-9.]+" | awk '{print $2}')
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
pcap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | head -1)
clk=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null | head -1)
ecc=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | head -1)
persist=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1)
torch_ver=$("$PY_BIN" -c "import torch;print(torch.__version__)" 2>/dev/null)
torch_cuda=$("$PY_BIN" -c "import torch;print(torch.version.cuda)" 2>/dev/null)
py_ver=$("$PY_BIN" -c "import sys;print('.'.join(map(str,sys.version_info[:3])))" 2>/dev/null)
os_name=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")
kernel=$(uname -r)
cpu=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')
cores=$(nproc 2>/dev/null)
ram=$(free -g 2>/dev/null | awk 'NR==2{print $2" GB"}')

if [ "$json" = "1" ]; then
python3 - "$gpu_name" "$gpu_mem" "$gpu_count" "$driver" "$cuda_smi" "$nvcc_ver" "$cc" \
           "$pcap" "$clk" "$ecc" "$persist" "$torch_ver" "$torch_cuda" "$py_ver" \
           "$os_name" "$kernel" "$cpu" "$cores" "$ram" <<'PY'
import json, sys
k = ["gpu","gpu_memory","gpu_count","driver","cuda_driver","nvcc","compute_capability",
     "power_limit","max_sm_clock","ecc","persistence_mode","torch","torch_cuda","python",
     "os","kernel","cpu","cpu_cores","ram"]
print(json.dumps(dict(zip(k, sys.argv[1:])), indent=2))
PY
else
  printf '%-22s %s\n' \
    "GPU"            "${gpu_count}x ${gpu_name} (${gpu_mem})" \
    "Compute cap"    "sm_${cc//./}" \
    "Driver"         "$driver" \
    "CUDA (driver)"  "$cuda_smi" \
    "CUDA (nvcc)"    "${nvcc_ver:-not installed}" \
    "Power limit"    "$pcap" \
    "Max SM clock"   "$clk" \
    "ECC"            "$ecc" \
    "Persistence"    "$persist" \
    "torch"          "${torch_ver:-n/a} (cuda ${torch_cuda:-n/a})" \
    "Python"         "$py_ver" \
    "OS"             "$os_name" \
    "Kernel"         "$kernel" \
    "CPU"            "$cpu ($cores cores)" \
    "RAM"            "$ram"
fi
