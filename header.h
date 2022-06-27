#ifndef HEADER_H
#define HEADER_H

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

using small_t = cutlass::int4b_t;
using large_t = int32_t;

template<class T> using HostTensorRow = cutlass::HostTensor<T, cutlass::layout::RowMajor>;
template<class T> using HostTensorCol = cutlass::HostTensor<T, cutlass::layout::ColumnMajor>;

template<class T> using TensorRefRow = cutlass::TensorRef<T, cutlass::layout::RowMajor>;
template<class T> using TensorRefCol = cutlass::TensorRef<T, cutlass::layout::ColumnMajor>;

// SQA parameters
#define N 4096
#define M 1024
#define M_2 128
#define G0 32.
#define incVal 4.

#define TIMES 1
#define STEP 1

// GPU parameter
#define NUM_GPU 2
#define NUM_STREAM 2
#define NUM_BLK 108
#define NUM_TRD 128
#define M_GPU (M / NUM_GPU)
#define N_GPU (N / NUM_GPU)
#define M_OK max(M_GPU, 32)

// Must be multiples of 16
#define MATRIX_M N
#define MATRIX_K M_2
#define MATRIX_N M

// Error check macros
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }


#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define CUTLASS_CHECK(status) { cutlass::Status error = status; if (error != cutlass::Status::kSuccess) { std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

void cudaErrCheck_(cudaError_t stat, const char *file, int line);

void initCoupling(std::vector<TensorRefRow<small_t>> &couplings_refs, 
                  char *file_name);

void construct_spin_CPU(HostTensorCol<small_t> &spin);

void construct_delta_H_CPU(std::vector<TensorRefRow<small_t>> &couplings_refs,
                           HostTensorCol<small_t>             &spin,
                           HostTensorCol<large_t>             &delta_H);

float calculate_maxcut(HostTensorCol<small_t> &spin, 
                       std::vector<TensorRefRow<small_t>> &couplings_refs);
#endif
