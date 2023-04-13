#include <cpu_utils.h>
#include <random>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <dataset.h>
#include <kernels.h>
#include <device_utils.h>
#include <consts.h>

KMeansResult Dataset::kmeans_openmp(uint32_t n_centroids, uint32_t max_iters) {

    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 
    
    CpuTimer timer;
    vector<float> time_per_iter;
    
    // kmeans iteration
    for(uint32_t iter = 0; iter < max_iters; iter++) {

        timer.start();

        compute_assignments(points, centroids, assignments, n_points, n_centroids, n_dims);
        recenter_centroids(points, centroids, assignments, n_points, n_centroids, n_dims);
        
        timer.stop();
        time_per_iter.push_back(timer.elapsed_time());
    }

    return KMeansResult(centroids, n_centroids, assignments, time_per_iter);
}

KMeansResult Dataset::kmeans_cuda(uint32_t n_centroids, uint32_t max_iters, bool fused_kernel) {

    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 

    CudaTimer timer;
    vector<float> time_per_iter;

    int blocks_assignment = n_points / (THREADS_PER_BLOCK * PTS_PER_THREAD);
    int blocks_accumulate = n_points * n_dims / (THREADS_PER_BLOCK * CALCS_PER_THREAD);
    int blocks_reduce_divide = n_centroids * n_dims / (THREADS_PER_BLOCK * CALCS_PER_THREAD);

    size_t shmem_size = (THREADS_PER_BLOCK * PTS_PER_THREAD * n_dims + SHM_K * SHM_DIM) * sizeof(float); 

    float *d_points, *d_centroids, *d_accumulator;
    uint32_t *d_assignments, *d_counts, *d_n_points, *d_n_centroids, *d_n_dims;
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
        &d_accumulator, &d_counts,
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );

    for(uint32_t iter = 0; iter < max_iters; iter++) {

        timer.start();

        if(fused_kernel) {
            fused_assignment_accumulate_kernel<<< blocks_assignment, THREADS_PER_BLOCK, shmem_size >>> (
                d_points, d_centroids, d_accumulator, d_assignments, d_counts, d_n_points, d_n_centroids, d_n_dims);
        } else {
            
            compute_assignments_kernel<<< blocks_assignment, THREADS_PER_BLOCK, shmem_size >>> (
                d_points, d_centroids, d_assignments, d_n_points, d_n_centroids, d_n_dims);

            accumulate_cluster_members_kernel<<< blocks_accumulate, THREADS_PER_BLOCK >>> (
                d_points, d_accumulator, d_assignments, d_counts, d_n_points, d_n_centroids, d_n_dims);
        }
        
        
        reduce_private_copies_kernel<<< blocks_reduce_divide, THREADS_PER_BLOCK >>>(d_accumulator, d_n_centroids, d_n_dims);
        divide_centroids_kernel<<< blocks_reduce_divide, THREADS_PER_BLOCK >>>(d_accumulator, d_counts, d_n_centroids, d_n_dims);

        cudaMemcpy(d_centroids, d_accumulator, n_centroids * n_dims * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d_counts, 0, n_centroids * sizeof(uint32_t));
        cudaMemset(d_accumulator, 0, n_centroids * n_dims * NUM_PRIV_COPIES * sizeof(float));

        timer.stop();
        time_per_iter.push_back(timer.elapsed_time());
    }

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
    
    return KMeansResult(centroids, n_centroids, assignments, time_per_iter);
}
