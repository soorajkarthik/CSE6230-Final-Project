#include <dataset.h>
#include <kernels.h>
#include <device_utils.h>
#include <consts.h>

KMeansResult Dataset::kmeans_cuda(uint32_t n_centroids, uint32_t max_iters, float tol) {
    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 

    vector<float> time_per_iter;

    int threads_per_block = 16;
    int blocks_assignment = n_points / (threads_per_block * PTS_PER_THREAD);
    
    int calcs_per_thread = 16;
    int blocks_accumulate = n_points * n_dims / (threads_per_block * calcs_per_thread);
    int blocks_reduce_divide = n_centroids * n_dims / (threads_per_block * calcs_per_thread);

    size_t shmem_size = (threads_per_block * PTS_PER_THREAD * n_dims + SHM_K * SHM_DIM) * sizeof(float); 

    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_sizes, *d_n_points, *d_n_centroids, *d_n_dims;
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_sizes,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    compute_assignments_kernel<<< blocks_assignment, threads_per_block, shmem_size >>> (d_points, d_centroids, d_assignments, d_n_points, d_n_centroids, d_n_dims);
    accumulate_cluster_members_kernel<<< blocks_accumulate, threads_per_block >>> (d_points, d_accumulator, d_assignments, d_sizes, d_n_points, d_n_centroids, d_n_dims);

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
    
    return KMeansResult(centroids, n_centroids, assignments, time_per_iter);
}
