#include <kmeans_cuda.h>
#include <dataset.h>
#include <cstdio>

void host_to_device_init_transfer(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims) {

    cudaMalloc(d_points,      n_points * n_dims * sizeof(float));
    cudaMalloc(d_centroids,   n_centroids * n_dims * sizeof(float));
    cudaMalloc(d_assignments, n_points * sizeof(uint32_t));
    cudaMalloc(d_n_points,    sizeof(uint32_t));
    cudaMalloc(d_n_centroids, sizeof(uint32_t));
    cudaMalloc(d_n_dims,      sizeof(uint32_t));

    cudaMemcpy(*d_points,      points,       n_points * n_dims * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(*d_centroids,   centroids,    n_centroids * n_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_assignments, assignments,  n_points * sizeof(uint32_t),          cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_points,    &n_points,    sizeof(uint32_t),                     cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_centroids, &n_centroids, sizeof(uint32_t),                     cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_dims,      &n_dims, 	    sizeof(uint32_t),                     cudaMemcpyHostToDevice);
}

void device_to_host_transfer_free(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims) {

    cudaMemcpy(assignments, *d_assignments, n_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(&d_points);
    cudaFree(&d_centroids);
    cudaFree(&d_assignments);
    cudaFree(&d_n_points);
    cudaFree(&d_n_centroids);
    cudaFree(&d_n_dims);
}

__global__ void compute_assignments_kernel(
    float *__restrict__ points, 
    float *__restrict__ centroids, 
    uint32_t *__restrict__ assignments, 
    uint32_t *__restrict__ n_points, 
    uint32_t *__restrict__ n_centroids, 
    uint32_t *__restrict__ n_dims) {

    uint32_t K = *n_centroids, D = *n_dims;

    extern __shared__ float shmem[];

    float *shm_points = shmem;
    float *shm_centroids = shm_points + PTS_PER_THREAD * D * blockDim.x; 
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int point_idx = tid * PTS_PER_THREAD;
    int point_offset = point_idx * D;

    int shm_point_idx = threadIdx.x * PTS_PER_THREAD;
    int shm_point_offset = shm_point_idx * D;

    float dists[PTS_PER_THREAD][SHM_K];
    float min_dists[PTS_PER_THREAD];
    uint32_t local_assignments[PTS_PER_THREAD];

    // Load points into shared memory
    #pragma unroll
    for(int i = 0; i < PTS_PER_THREAD; i++) {
        for(int d = 0; d < D; d++) {
            shm_points[shm_point_offset + i * D + d] = points[point_offset + i * D + d];
        }
        min_dists[i] = 1e30;
    }

    // Tiled loop over K
    for(int k_block = 0; k_block < K; k_block += SHM_K) {

        // Clear distances
        #pragma unroll
        for(int i = 0; i < PTS_PER_THREAD; i++) {

            #pragma unroll
            for(int j = 0; j < SHM_K; j++) {
                dists[i][j] = 0;
            }
        }

        // Tiled loop over D
        for(int d_block = 0; d_block < D; d_block += SHM_DIM) {

            // Load centroids into shared memory
            int k, d, p;
            for(int shm_idx = threadIdx.x; shm_idx < SHM_K * SHM_DIM; shm_idx += blockDim.x) {
                
                k = shm_idx / SHM_DIM;
                d = shm_idx % SHM_DIM;


                shm_centroids[shm_idx] = centroids[(k + k_block) * D + (d + d_block)];
            }

            __syncthreads();

            // Accumulate distances for this set of dimensions
            #pragma unroll
            for(k = 0; k < SHM_K; k++) {

                #pragma unroll
                for(d = 0; d < SHM_DIM; d++) {

                    float centroid_val = shm_centroids[k * SHM_DIM + d];

                    #pragma unroll
                    for(p = 0; p < PTS_PER_THREAD; p++) {
                        float val = centroid_val - shm_points[shm_point_offset + p * D + (d + d_block)];
                        dists[p][k] += val * val;
                    }
                }
            }

            __syncthreads();
        }


        // Reassign
        #pragma unroll
        for(int k = 0; k < SHM_K; k++) {

            #pragma unroll
            for(int p = 0; p < PTS_PER_THREAD; p++) {

                if(min_dists[p] > dists[p][k]) {
                    min_dists[p] = dists[p][k];
                    local_assignments[p] = k + k_block;
                }
            }
        }
    }

    // Write final assignments to global memory
    #pragma unroll
    for(int p = 0; p < PTS_PER_THREAD; p++) {
        assignments[point_idx + p] = local_assignments[p];
    }
}


KMeansResult Dataset::kmeans_cuda(uint32_t n_centroids, uint32_t max_iters, float tol) {
    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 

    vector<float> time_per_iter;

    int threads_per_block = 5;
    int blocks = n_points / (threads_per_block * PTS_PER_THREAD);
    size_t shmem_size = (threads_per_block * PTS_PER_THREAD * n_dims + SHM_K * SHM_DIM) * sizeof(float); 

    float *d_points, *d_centroids;
    uint32_t *d_assignments, *d_n_points, *d_n_centroids, *d_n_dims;
    
    host_to_device_init_transfer(
        points, &d_points,
        centroids, &d_centroids,
        assignments, &d_assignments,
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
        n_points, &d_n_points,
        n_centroids, &d_n_centroids,
        n_dims, &d_n_dims
    );
    
    return KMeansResult(centroids, n_centroids, assignments, time_per_iter);
}
