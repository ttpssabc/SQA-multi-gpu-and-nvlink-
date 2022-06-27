#include <stdio.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/numeric_conversion.h>

#include "header.h"

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

void initCoupling(std::vector<TensorRefRow<small_t>> &couplings_refs, 
                  char *file_name)
{
    FILE *instance = fopen(file_name, "r");
    assert(instance != NULL);
    int a, b, total_spins, total_couplings;
    int w;
    fscanf(instance, "%d%d", &total_spins, &total_couplings);

    for (auto &ref:couplings_refs)
        memset(ref.data(), 0, N * M_2 * cutlass::sizeof_bits<small_t>::value / 8);
        
    while (total_couplings--)
    {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        a--;
        b--;
        couplings_refs[a / M_2].at({b, a % M_2}) = small_t(-w);
        couplings_refs[b / M_2].at({a, b % M_2}) = small_t(-w);
    }
    fclose(instance);
}

void construct_spin_CPU(HostTensorCol<small_t> &spin)
{
    #pragma omp parallel for
    for (int n = 0; n < N; n++){
        for(int m = 0; m < M_GPU; m++){
            float x = ((float)rand()/(float)(RAND_MAX)) * 1.0;    
            spin.host_ref().at({n, m}) = (x>=0.5) ? small_t(1) : small_t(-1);
        }
    }

    spin.sync_device();
}

void construct_delta_H_CPU(std::vector<TensorRefRow<small_t>> &couplings_refs,
                           HostTensorCol<small_t>             &spin,
                           HostTensorCol<large_t>             &delta_H)
{

    memset(delta_H.host_data(), 0, delta_H.size() * cutlass::sizeof_bits<large_t>::value / 8);

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        for(int m = 0; m < M_GPU; m++) {
            for (int i = 0; i < N; ++i) {
                delta_H.host_ref().at({n, m}) += couplings_refs[n / M_2].at({i, n % M_2}).get() * spin.host_ref().at({i, m}).get();
            }
        }
    }

    delta_H.sync_device();
}

float calculate_maxcut(HostTensorCol<small_t> &spin, 
                       std::vector<TensorRefRow<small_t>> &couplings_refs)
{
    spin.sync_host();
    
    cutlass::NumericConverter<small_t, float> converter;

    float maxCut = 0.;

    #pragma omp parallel for schedule(dynamic) reduction(+:maxCut) 
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            if (spin.host_ref().at({i, 0}).get() != spin.host_ref().at({j, 0}).get())
            {
                maxCut += converter(couplings_refs[i / M_2].at({j, i % M_2}).get());
            }
        }
    }
    
    return -(maxCut);
}