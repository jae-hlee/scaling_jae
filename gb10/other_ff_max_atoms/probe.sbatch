#!/bin/bash
#SBATCH --job-name=ffoprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/data/jlee859/ffother/logs/probe_%j.out

# What a GB10 node has for building GPUMD (NEP), LAMMPS + Kokkos (SNAP,
# Allegro) and DeePMD-kit (DP): compilers, CUDA toolkit, MPI, CMake, the
# Python env, internet access, disk space, the training data. Changes nothing
# except a scratch folder that it removes. About a minute.
#   mkdir -p /data/jlee859/ffother/logs
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 probe.sbatch
set -u
B=/data/jlee859/ffother
CONDA=/data/jlee859/miniforge3-aarch64
PY=$CONDA/envs/alignn/bin/python
sec() { echo; echo "== $1"; }
try() { echo "\$ $*"; timeout 60 "$@" 2>&1 | head -n "${LINES_MAX:-12}"; }

sec "node"
echo "$(date) | $(hostname) | $(uname -m) | kernel $(uname -r)"
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
echo "cores: $(nproc)"
free -g | head -n 2
df -h /data/jlee859 /tmp 2>&1
echo "glibc: $(ldd --version 2>&1 | head -n 1)"

sec "GPU"
try nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
try nvidia-smi --query-compute-apps=pid,used_memory --format=csv
LINES_MAX=25 try nvidia-smi

sec "compilers and build tools on the node"
for c in gcc g++ gfortran cc c++ clang make cmake ninja git wget curl tar unzip; do
    p=$(command -v $c 2>/dev/null) || { echo "$c: NOT FOUND"; continue; }
    echo "$c: $p | $($c --version 2>&1 | head -n 1)"
done

sec "CUDA toolkit (nvcc, headers, libraries)"
echo "CUDA_HOME=${CUDA_HOME:-unset} CUDA_PATH=${CUDA_PATH:-unset}"
command -v nvcc && nvcc --version | tail -n 2
ls -d /usr/local/cuda* /opt/cuda* /usr/lib/cuda* 2>/dev/null
for d in /usr/local/cuda*/bin /usr/bin; do
    [ -x $d/nvcc ] && echo "nvcc at $d/nvcc: $($d/nvcc --version | tail -n 1)"
done
find /usr/local /usr/include /opt -maxdepth 5 -name cuda_runtime.h 2>/dev/null | head -n 5
ldconfig -p 2>/dev/null | grep -E 'libcudart|libcublas\.|libcusolver\.|libcufft\.|libcurand\.|libnvrtc\.' | head -n 12

sec "MPI and environment modules"
for c in mpirun mpiexec mpicc mpicxx srun; do
    p=$(command -v $c 2>/dev/null) && echo "$c: $p" || echo "$c: NOT FOUND"
done
command -v mpirun >/dev/null && mpirun --version 2>&1 | head -n 2
type module >/dev/null 2>&1 && { module avail 2>&1 | head -n 30; } || echo "no environment modules"

sec "conda"
ls $CONDA/bin 2>/dev/null | grep -E '^(conda|mamba)$'
ls $CONDA/envs 2>/dev/null
du -sh $CONDA 2>/dev/null

sec "python env 'alignn'"
$PY - <<'EOF'
import sys, platform
print("python", sys.version.split()[0], platform.machine())
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0),
      "| arch list", torch.cuda.get_arch_list())
print("cxx11 abi", torch.compiled_with_cxx11_abi())
print("cmake prefix", torch.utils.cmake_prefix_path)
import os
lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print("torch/lib:", ", ".join(sorted(f for f in os.listdir(lib)
                                      if f.startswith(("libtorch", "libc10")))))
free, total = torch.cuda.mem_get_info()
print(f"cuda mem_get_info: free {free / 2**30:.1f} GiB, total {total / 2**30:.1f} GiB")
EOF
$PY -m pip list 2>/dev/null | grep -i -E '^(numpy|scipy|ase|matscipy|vesin|e3nn|lightning|pytorch-lightning|hydra-core|torchmetrics|nequip|allegro|deepmd-kit|dpdata|fitsnap3|lammps|mpi4py|cmake|ninja|scikit-build|pandas|jarvis-tools|dgl|nvidia-[a-z0-9-]+)\s'
$PY -m pip --version
$PY -c "import sysconfig; print('python include', sysconfig.get_paths()['include'])"

sec "internet from this node"
for u in https://github.com https://pypi.org/simple/pip/ https://download.lammps.org https://conda.anaconda.org/conda-forge/linux-aarch64/repodata.json.zst; do
    echo "$u -> $(curl -s -o /dev/null -m 20 -I -w '%{http_code}' $u 2>&1)"
done
try git ls-remote https://github.com/lammps/lammps.git HEAD

sec "a CUDA program compiles and runs"
S=$B/probe_scratch_$SLURM_JOB_ID
mkdir -p $S && cd $S
cat > t.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int* a) { a[threadIdx.x] = threadIdx.x; }
int main() {
  int* d; int h[8];
  if (cudaMalloc(&d, sizeof(h)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
  k<<<1, 8>>>(d);
  cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  size_t f, t; cudaMemGetInfo(&f, &t);
  printf("kernel ok %d | cudaMemGetInfo free %.1f GiB total %.1f GiB\n",
         h[7], f / 1073741824.0, t / 1073741824.0);
  return h[7] == 7 ? 0 : 1;
}
EOF
NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -n 1)
if [ -n "$NVCC" ]; then
    for arch in "" "-arch=sm_121" "-arch=sm_120" "-arch=sm_75" "-arch=sm_60"; do
        rm -f t.out
        if $NVCC -O2 $arch t.cu -o t.out > build.log 2>&1; then
            echo "nvcc ${arch:-(default arch)}: builds | $(./t.out 2>&1 | head -n 1)"
        else
            echo "nvcc ${arch:-(default arch)}: FAILS | $(grep -m 1 -i -E 'error|unsupported|invalid' build.log)"
        fi
    done
else
    echo "no nvcc found: GPUMD and Kokkos need the CUDA toolkit"
fi
cd $B && rm -rf $S

sec "training data (MatPES r2SCAN, as used for stage 12)"
find /data/jlee859/ffsmooth/data -maxdepth 2 -name 'id_prop.json*' -exec ls -la {} \; 2>&1 | head -n 5

echo
echo "probe end $(date)"
