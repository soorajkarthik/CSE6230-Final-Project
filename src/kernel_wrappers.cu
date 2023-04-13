#include <kernel_wrappers.h>
#include <kmeans_cuda.h>

void call_compute_assignments_kernel(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {
    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_sizes, *d_n_points, *d_n_centroids, *d_n_dims;

    int threads_per_block = 16;
    int blocks = n_points / (threads_per_block * PTS_PER_THREAD);
    size_t shmem_size = (threads_per_block * PTS_PER_THREAD * n_dims + SHM_K * SHM_DIM) * sizeof(float); 

    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_sizes,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    compute_assignments_kernel<<< blocks, threads_per_block, shmem_size >>> (d_points, d_centroids, d_assignments, d_n_points, d_n_centroids, d_n_dims);

    cudaDeviceSynchronize();

    device_to_host_transfer_free(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_sizes,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
}

void call_recenter_centroids_kernels(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {
    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_sizes, *d_n_points, *d_n_centroids, *d_n_dims;

    int threads_per_block = 16;
    int calcs_per_thread = 16;
    int blocks_accumulate = n_points * n_dims / (threads_per_block * calcs_per_thread);
    int blocks_reduce_divide = n_centroids * n_dims / (threads_per_block * calcs_per_thread);
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_sizes,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    cudaMemcpy(d_assignments, assignments, n_points * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_accumulator, 0, n_centroids * n_dims * NUM_PRIV_COPIES * sizeof(float));
    cudaMemset(d_sizes, 0, n_centroids * sizeof(uint32_t));

    accumulate_cluster_members_kernel<<<blocks_accumulate, threads_per_block >>>(d_points, d_accumulator, d_assignments, d_sizes, d_n_points, d_n_centroids, d_n_dims);
    cudaDeviceSynchronize();

    reduce_private_copies_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_n_centroids, d_n_dims);
    cudaDeviceSynchronize();
    
    divide_centroids_kernel<<< blocks_reduce_divide, threads_per_block >>>(d_accumulator, d_sizes, d_n_centroids, d_n_dims);
    cudaDeviceSynchronize();

    cudaMemcpy(d_centroids, d_accumulator, n_centroids * n_dims * sizeof(float), cudaMemcpyDeviceToDevice);

    device_to_host_transfer_free(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_sizes,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
}