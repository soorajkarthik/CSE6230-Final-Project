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