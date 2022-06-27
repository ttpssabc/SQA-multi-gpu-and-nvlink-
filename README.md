### Required Library

[cutlass](https://github.com/NVIDIA/cutlass)

### How to compile
```
nvcc main.cu kernel.cu utils.cu \
-arch={GPU_arch} \
-Xcompiler \
-fopenmp  \
-I{CUTLASS_PATH}/include \
-I{CUTLASS_PATH}/build/include \
-I{CUTLASS_PATH}/tools/util/include \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
-DTEST_MAXCUT
```
GPU_arch: \
&nbsp;&nbsp;&nbsp;&nbsp;The architecture of GPU, for example, sm_75 for 2080ti \
CUTLASS_PATH: \
&nbsp;&nbsp;&nbsp;&nbsp;Where your cutlass library is installed. \
-DTEST_MAXCUT: \
&nbsp;&nbsp;&nbsp;&nbsp;Specify this flag if you want to test maxcut problem. \
&nbsp;&nbsp;&nbsp;&nbsp;If you only want to test time or memory, it will run faster without the flag (skip local-field initilization)
