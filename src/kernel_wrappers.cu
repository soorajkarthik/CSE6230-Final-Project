#include <kernel_wrappers.h>
#include <device_utils.h>
#include <kernels.h>
#include <consts.h>

void call_compute_assignments_kernel(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {
    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_counts, *d_n_points, *d_n_centroids, *d_n_dims;

    int threads_per_block = 16;
    int blocks = n_points / (threads_per_block);

    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    compute_assignments_kernel<<< blocks, threads_per_block >>> (d_points, d_centroids, d_assignments, d_n_points, d_n_centroids, d_n_dims);

    cudaDeviceSynchronize();

    device_to_host_transfer_free(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
}

void call_recenter_centroids_kernels(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {
    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_counts, *d_n_points, *d_n_centroids, *d_n_dims;

    int threads_per_block = 16;
    int calcs_per_thread = 16;
    int blocks_accumulate = n_points * n_dims / (threads_per_block * calcs_per_thread);
    int blocks_reduce_divide = n_centroids * n_dims / (threads_per_block * calcs_per_thread);
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    cudaMemcpy(d_assignments, assignments, n_points * sizeof(uint32_t), cudaMemcpyHostToDevice);

    accumulate_cluster_members_kernel<<<blocks_accumulate, threads_per_block >>>(d_points, d_accumulator, d_assignments, d_counts, d_n_points, d_n_centroids, d_n_dims);
    reduce_private_copies_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_n_centroids, d_n_dims);
    divide_centroids_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_counts, d_n_centroids, d_n_dims);

    cudaMemcpy(d_centroids, d_accumulator, n_centroids * n_dims * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    device_to_host_transfer_free(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
}

void call_fused_assignment_recenter_kernels(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {
    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_counts, *d_n_points, *d_n_centroids, *d_n_dims;


    int threads_per_block = 16;

    int blocks_fused = n_points / (threads_per_block);

    int calcs_per_thread = 16;
    int blocks_reduce_divide = n_centroids * n_dims / (threads_per_block * calcs_per_thread);
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    fused_assignment_accumulate_kernel<<< blocks_fused, threads_per_block >>>(d_points, d_centroids, d_accumulator, d_assignments, d_counts, d_n_points, d_n_centroids, d_n_dims);
    reduce_private_copies_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_n_centroids, d_n_dims);
    divide_centroids_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_counts, d_n_centroids, d_n_dims);

    cudaMemcpy(d_centroids, d_accumulator, n_centroids * n_dims * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaDeviceSynchronize();
    
    device_to_host_transfer_free(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
}