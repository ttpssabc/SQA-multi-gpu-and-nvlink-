#include <stdio.h>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/device/tensor_fill.h>

#include "header.h"
#include "kernel.h"

__global__ void judge_flipping_com(TensorRefRow<small_t> couplings_ref,
                                   TensorRefCol<large_t> delta_H_ref,
                                   TensorRefCol<small_t> spin_ref,
                                   TensorRefCol<small_t> matrix_B_ref,
                                   TensorRefCol<float>   log_rand_val_ref,
                                   TensorRefCol<small_t> upper_spins_ref,
                                   TensorRefCol<small_t> lower_spins_ref,
                                   float J_perp, float beta, int start_spin)
{
    int m = blockIdx.x;
    cutlass::layout::ColumnMajor::TensorCoord idx;
    cutlass::layout::ColumnMajor::TensorCoord mb_idx;
    int upper, lower;
    float delta;
    int first_rd_idx = m & 1; // even:0, odd:1

    extern __shared__ large_t deltas[];
    deltas[threadIdx.x] = delta_H_ref.at({start_spin + threadIdx.x, m});

    upper = (m - 1) & (M_GPU - 1);
    lower = (m + 1) & (M_GPU - 1);

    for (int n = 0; n < M_2; n++)
    {
        int nn = start_spin + ((first_rd_idx * (M_2 / 2) + n) & (M_2 - 1));
        int nn_ = ((first_rd_idx * (M_2 / 2) + n) & (M_2 - 1));

        idx = {nn, m};
        mb_idx = {nn & (M_2 - 1), m};
        delta = (float)deltas[nn & (M_2 - 1)];

        float spin_upper = (float) (m == 0)         ? upper_spins_ref.at({nn, 1}).get() : spin_ref.at({nn, upper}).get();
        float spin_lower = (float) (m == M_GPU - 1) ? lower_spins_ref.at({nn, 1}).get() : spin_ref.at({nn, lower}).get();

        delta = beta * (float)spin_ref.at(idx).get() * (delta - J_perp * (spin_upper + spin_lower));

        matrix_B_ref.at(mb_idx) = small_t(0);
        if ((log_rand_val_ref.at(idx)) > delta)
        {
            if (threadIdx.x == 0)
            {
                small_t orig_spin = spin_ref.at(idx).get();
                spin_ref.at(idx) = -orig_spin;
                matrix_B_ref.at(mb_idx) = -(orig_spin + orig_spin);
            }
            __syncthreads();
            int ii = start_spin + threadIdx.x;
            deltas[threadIdx.x] += (large_t)(couplings_ref.at({ii, nn_}).get() * matrix_B_ref.at(mb_idx).get());
        }
        __syncthreads();
    }
}

void update_delta_H_cutlass(TensorRefRow<small_t>  &couplings_dev_ref, 
                            HostTensorCol<small_t> &matrix_B, 
                            HostTensorCol<large_t> &delta_H,
                            int which_spin,
                            cudaStream_t stream)
{

    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor, 
        cutlass::int4b_t, cutlass::layout::ColumnMajor, 
        ElementOutput, cutlass::layout::ColumnMajor,
        ElementAccumulator, 
        cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>, 
        cutlass::gemm::GemmShape<8, 8, 32>,
        cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
        2>;

    int split_k_slices = 1;

    typename Gemm::Arguments arguments{{N, M_OK, M_2},
                                       couplings_dev_ref,
                                       matrix_B.device_ref(),
                                       delta_H.device_ref(),
                                       delta_H.device_ref(),
                                       {1, 1}, 
                                       split_k_slices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op(stream);
    CUTLASS_CHECK(status);    
}

void construct_delta_H(HostTensorRow<small_t> &couplings,
                       HostTensorCol<small_t> &spin,
                       HostTensorCol<large_t> &delta_H)
{
    cutlass::reference::device::TensorFill(delta_H.device_view());

    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor, 
        cutlass::int4b_t, cutlass::layout::ColumnMajor, 
        ElementOutput, cutlass::layout::ColumnMajor,
        ElementAccumulator, 
        cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>, 
        cutlass::gemm::GemmShape<8, 8, 32>,
        cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
        2>;

    int split_k_slices = 1;

    typename Gemm::Arguments arguments{{N, M_OK, N},
                                       couplings.device_ref(),
                                       spin.device_ref(),
                                       delta_H.device_ref(),
                                       delta_H.device_ref(),
                                       {1, 1},
                                       split_k_slices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);
}

__global__ void construct_lograndval(float *log_rand_val_dev, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    curand_init(idx, idx, 0, &states[idx]);

    for (int i = idx; i < M_GPU * N; i += stride)
    {
        log_rand_val_dev[i] = -log(curand_uniform(&states[idx]));
    }
}

__global__ void construct_spin(TensorRefCol<small_t> spin_ref, curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    curand_init(idx, idx, 0, &states[idx]);

    for (int n = 0; n < N; ++n)
    {
        for (int m = idx; m < M_GPU; m += stride)
        {
            spin_ref.at({n, m}) = (curand_uniform(&states[idx]) > 0.5) ? small_t(1) : small_t(-1);
        }
    }
}
