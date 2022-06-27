#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

#include "header.h"
#include "kernel.h"

int main(int argc, char *argv[])
{
    /* Check command line arguments */
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [coupling file]\n", argv[0]);
        exit(-1);
    }

    /* Coupling on host */
    std::vector<small_t*>                                               couplings_ptrs(N / M_2);
    std::vector<cutlass::TensorRef<small_t, cutlass::layout::RowMajor>> couplings_refs(N / M_2);

    for (auto i = 0; i < couplings_ptrs.size(); ++i) {
        cudaErrCheck(cudaMallocHost((void **)&couplings_ptrs[i], N * M_2 * cutlass::sizeof_bits<small_t>::value / 8));
        couplings_refs[i] = cutlass::TensorRef<small_t, cutlass::layout::RowMajor>(couplings_ptrs[i], M_2);
    }
    
    /* Init Couplings */  
    initCoupling(couplings_refs, argv[1]);

    /* Shared data between multi-threads */
    std::vector<small_t*>                upper_spins_dev_ptrs(NUM_GPU);
    std::vector<small_t*>                lower_spins_dev_ptrs(NUM_GPU);
    std::vector<std::vector<small_t*>> couplings_dev_ptrs_vec(NUM_GPU);
    
    /* Events for synchronizing devices */
    cudaEvent_t event_memcpy[NUM_GPU][N_GPU / M_2 + 1];
    cudaEvent_t event_kernel[NUM_GPU][N_GPU / M_2 + 1];
    cudaEvent_t   event_spin[NUM_GPU];

    omp_set_nested(1);

#pragma omp parallel num_threads(NUM_GPU) shared(couplings_refs, upper_spins_dev_ptrs, lower_spins_dev_ptrs, \
                                                 couplings_dev_ptrs_vec, \
                                                 event_memcpy, event_kernel, event_spin)
{
    /* Get thread id and set corresponding GPU */
    int id = omp_get_thread_num();
    int upper_id = (id + NUM_GPU - 1) % NUM_GPU;
    int lower_id = (id + NUM_GPU + 1) % NUM_GPU;
    cudaErrCheck(cudaSetDevice(id));

    /* Enable nvlink */
    cudaDeviceEnablePeerAccess(upper_id, 0);

    #if NUM_GPU > 2
    cudaDeviceEnablePeerAccess(lower_id, 0);
    #endif

    /* Create event on current device */
    for (auto i = 0; i < N_GPU / M_2 + 1; ++i) cudaEventCreate(&event_memcpy[id][i]);
    for (auto i = 0; i < N_GPU / M_2 + 1; ++i) cudaEventCreate(&event_kernel[id][i]);
    cudaEventCreate(&event_spin[id]);

    /* Private device data */
    cutlass::HostTensor<small_t, cutlass::layout::ColumnMajor>         spin({  N,  M_OK});
    cutlass::HostTensor<large_t, cutlass::layout::ColumnMajor>      delta_H({  N,  M_OK});
    cutlass::HostTensor<small_t, cutlass::layout::ColumnMajor>     matrix_B({M_2,  M_OK});
    cutlass::HostTensor<  float, cutlass::layout::ColumnMajor> log_rand_val({  N, M_GPU});

    /* Coupling on device (N_GPU / M_2 + 1) */        
    std::vector<small_t*>                                               couplings_dev_ptrs(N_GPU / M_2 + 1);
    std::vector<cutlass::TensorRef<small_t, cutlass::layout::RowMajor>> couplings_dev_refs(N_GPU / M_2 + 1);

    for (auto i = 0; i < couplings_dev_ptrs.size(); ++i) {
        cudaErrCheck(cudaMalloc((void **)&couplings_dev_ptrs[i], N * M_2 * cutlass::sizeof_bits<small_t>::value / 8));
        couplings_dev_refs[i] = cutlass::TensorRef<small_t, cutlass::layout::RowMajor>(couplings_dev_ptrs[i], M_2);
    }

    /* Set the shared coupling device pointer vector */
    couplings_dev_ptrs_vec[id] = couplings_dev_ptrs;

    /* Initialize the couplings on device */
    #ifdef TEST_MAXCUT
    for (auto n = 0; n < N_GPU; n += M_2)
    {
        auto start_spin = id * N_GPU + n;

        cudaMemcpy(couplings_dev_refs[         n / M_2].data(),
                       couplings_refs[start_spin / M_2].data(), 
                   N * M_2 * cutlass::sizeof_bits<small_t>::value / 8, 
                   cudaMemcpyHostToDevice);
    }
    #endif

    /* Set the shared upper / lower spins */
    cudaErrCheck(cudaMalloc((void **)&upper_spins_dev_ptrs[id], N * cutlass::sizeof_bits<small_t>::value / 8));
    cudaErrCheck(cudaMalloc((void **)&lower_spins_dev_ptrs[id], N * cutlass::sizeof_bits<small_t>::value / 8));
    cutlass::TensorRef<small_t, cutlass::layout::ColumnMajor> upper_spins_dev_ref(upper_spins_dev_ptrs[id], N);
    cutlass::TensorRef<small_t, cutlass::layout::ColumnMajor> lower_spins_dev_ref(lower_spins_dev_ptrs[id], N);

    /* Init cuda streams */
    cudaStream_t streams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; ++i) 
        cudaErrCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    /* Init curandState */
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NUM_BLK * NUM_TRD * sizeof(curandState));

    /* Init parameters */
    float results[TIMES]   = {0.};
    float used_time[TIMES] = {0.};
    float increase = (incVal - 1/(float)incVal) / (float)STEP;
    float best_cut = -1e9;

    /* For measuring time */
    struct timeval begin, end;

    /* Current position for communication */
    auto src = 0;
    auto dst = (N_GPU / M_2);

    #pragma omp barrier

    /* SQA */
    for (int t = 0; t < TIMES; t++) {

        float beta = 1 / (float)incVal;

        /* Init spin and delta_H */
        #ifdef TEST_MAXCUT
        construct_spin_CPU(spin);
        construct_delta_H_CPU(couplings_refs, spin, delta_H);
        #endif

        gettimeofday(&begin, NULL);
        
        for (int p = 0; p < STEP; p++)
        {
            float Gamma  = G0 * (1. - (float) p / (float) STEP);
            float J_perp = -M * 0.5 * log(tanh((Gamma / M) * beta)) / beta;

            construct_lograndval<<<NUM_BLK, NUM_TRD, 0, streams[1]>>>(log_rand_val.device_data(), dev_random);
            
            /*
             *   
             *  | GPU0 |      |      |      |         |      | GPU0 |      |      |
             *  |      | GPU1 |      |      |   ->    |      |      | GPU1 |      |
             *  |      |      | GPU2 |      |         |      |      |      | GPU2 |
             *  |      |      |      | GPU3 |         | GPU3 |      |      |      |
             */
            for (int d = 0; d < NUM_GPU; ++d)
            {
                int blk_id = (d + id) % NUM_GPU;

                cudaMemcpyAsync(upper_spins_dev_ptrs[lower_id],
                                spin.device_data() + N * (M_GPU - 1) * cutlass::sizeof_bits<small_t>::value / 8,
                                N * cutlass::sizeof_bits<small_t>::value / 8,
                                cudaMemcpyDeviceToDevice,
                                streams[0]);

                cudaMemcpyAsync(lower_spins_dev_ptrs[upper_id],
                                spin.device_data(),
                                N * cutlass::sizeof_bits<small_t>::value / 8,
                                cudaMemcpyDeviceToDevice,
                                streams[0]);

                cudaEventRecord(event_spin[id], streams[0]);
                
                #pragma omp barrier
                cudaStreamWaitEvent(streams[0], event_spin[upper_id]);
                cudaStreamWaitEvent(streams[1], event_spin[upper_id]);

                #if NUM_GPU > 2
                cudaStreamWaitEvent(streams[0], event_spin[lower_id]);
                cudaStreamWaitEvent(streams[1], event_spin[lower_id]);
                #endif

                for (int n = 0; n < N_GPU; n += M_2)
                {

                    int start_spin = blk_id * N_GPU + n;

                    judge_flipping_com<<<M_GPU, M_2, M_2 * sizeof(large_t), streams[1]>>>(    couplings_dev_refs[src], 
                                                                                                 delta_H.device_ref(),
                                                                                                    spin.device_ref(), 
                                                                                                matrix_B.device_ref(),
                                                                                            log_rand_val.device_ref(),
                                                                                                  upper_spins_dev_ref,
                                                                                                  lower_spins_dev_ref,
                                                                                                               J_perp, 
                                                                                                         2 * M * beta, 
                                                                                                           start_spin);

                    update_delta_H_cutlass(couplings_dev_refs[src], matrix_B, delta_H, n, streams[1]);

                    cudaMemcpyAsync(couplings_dev_ptrs_vec[upper_id][dst],
                                    couplings_dev_ptrs_vec[      id][src], 
                                    N * M_2 * cutlass::sizeof_bits<small_t>::value / 8, 
                                    cudaMemcpyDeviceToDevice,
                                    streams[0]); 

                    cudaEventRecord(event_memcpy[id][dst], streams[0]);
                    cudaEventRecord(event_kernel[id][src], streams[1]);

                    #pragma omp barrier
                    cudaStreamWaitEvent(streams[0], event_memcpy[upper_id][dst]);
                    cudaStreamWaitEvent(streams[0], event_kernel[upper_id][src]);
                    cudaStreamWaitEvent(streams[1], event_memcpy[upper_id][dst]);
                    cudaStreamWaitEvent(streams[1], event_kernel[upper_id][src]);

                    src = (src + 1) % (N_GPU / M_2 + 1);
                    dst = (dst + 1) % (N_GPU / M_2 + 1);
                }
            }

            beta += increase;
        }
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        double duration = ((end.tv_sec - begin.tv_sec) * 1000000u + end.tv_usec - begin.tv_usec) / 1.e6;

        used_time[t] = duration;

        #ifdef TEST_MAXCUT
        results[t] = calculate_maxcut(spin, couplings_refs);
        #endif

        if(results[t] > best_cut) 
            best_cut = results[t];
    }

    /* Finalizing */
    if (id == 0)
    {
        float tot_result_time = 0.;
        float tot_energy      = 0.;
        for (int i = 0; i < TIMES; ++i) {
            tot_result_time += used_time[i];
            tot_energy      += results[i];
        }

        size_t freeMem, totalMem; 
        cudaMemGetInfo(&freeMem, &totalMem);

        /* Print Results */
        for (int t = 0; t < TIMES; t++)
            printf("TIME: %d,  used time (s): %10lf,  Maxcut: %10lf\n", t, used_time[t], results[t]);
        printf("\n");
        printf("Avg time  : %f\n", tot_result_time / TIMES);
        printf("Avg Maxcut: %f, Best cut = %f\n", tot_energy / TIMES, best_cut);
        printf("Memory: %zu\n", totalMem - freeMem);
    }

    for (int i = 0; i < NUM_STREAM; ++i)
        cudaStreamDestroy(streams[i]);
 
}

    return 0;
}