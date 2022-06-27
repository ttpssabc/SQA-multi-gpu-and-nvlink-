#ifndef KERNEL_H
#define KERNEL_H


__global__ void judge_flipping_com(TensorRefRow<small_t> couplings_ref,
                                   TensorRefCol<large_t> delta_H_ref,
                                   TensorRefCol<small_t> spin_ref,
                                   TensorRefCol<small_t> matrix_B_ref,
                                   TensorRefCol<float>   log_rand_val_ref,
                                   TensorRefCol<small_t> upper_spins_ref,
                                   TensorRefCol<small_t> lower_spins_ref,
                                   float J_perp, float beta, int start_spin);

void update_delta_H_cutlass(TensorRefRow<small_t>  &couplings_dev_ref, 
                            HostTensorCol<small_t> &matrix_B, 
                            HostTensorCol<large_t> &delta_H,
                            int which_spin,
                            cudaStream_t stream);

void construct_delta_H(HostTensorRow<small_t> &couplings,
                       HostTensorCol<small_t> &spin,
                       HostTensorCol<large_t> &delta_H);
                       
__global__ void construct_lograndval(float *log_rand_val_dev, curandState *states);

__global__ void construct_spin(TensorRefCol<small_t> spin_ref, curandState *states);

#endif