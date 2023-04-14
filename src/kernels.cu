#include <kernels.h>
#include <consts.h>

__global__ void compute_assignments_kernel(
    float const *__restrict__ points,
    float const *__restrict__ centroids,
    uint32_t *__restrict__ assignments,
    uint32_t const *__restrict__ n_points,
    uint32_t const *__restrict__ n_centroids,
    uint32_t const *__restrict__ n_dims) {

    uint32_t K = *n_centroids, D = *n_dims;

    volatile __shared__ float shm_centroids[SHM_K][SHM_DIM];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int point_offset = tid * D;

    float dists[SHM_K];
    float min_dist = 1e30;
    uint32_t local_assignment;
    int k, d;

    // Tiled loop over K
    for(int k_block = 0; k_block < K; k_block += SHM_K) {

        // Clear distances
        #pragma unroll
        for(int i = 0; i < SHM_K; i++) {
            dists[i] = 0;
        }

        // Tiled loop over D
        for(int d_block = 0; d_block < D; d_block += SHM_DIM) {

            // Load centroids into shared memory
            for(int shm_idx = threadIdx.x; shm_idx < SHM_K * SHM_DIM; shm_idx += blockDim.x) {
                
                k = shm_idx / SHM_DIM;
                d = shm_idx % SHM_DIM;


                shm_centroids[k][d] = centroids[(k + k_block) * D + (d + d_block)];
            }

            __syncthreads();

            // Accumulate distances for this set of dimensions
            #pragma unroll
            for(k = 0; k < SHM_K; k++) {

                #pragma unroll
                for(d = 0; d < SHM_DIM; d++) {

                    float centroid_val = shm_centroids[k][d];
                    float val = centroid_val - points[point_offset + (d + d_block)];
                    dists[k] += val * val;
                }
            }

            __syncthreads();
        }

        // Reassign
        #pragma unroll
        for(k = 0; k < SHM_K; k++) {
            if(min_dist > dists[k]) {
                min_dist = dists[k];
                local_assignment = k + k_block;
            }
        }
    }

    // Write final assignment to global memory
    assignments[tid] = local_assignment;
}

__device__ float* get_privatized_pointer(
    float *ptr, 
    uint32_t n_vecs, 
    uint32_t n_dims) {
    
    float *res = ptr;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / warpSize;
    
    // All processors in the same warp assigned to the same copy
    // since they write to consecutive elements
    res += (n_vecs * n_dims) * (wid % NUM_PRIV_COPIES);

    return res;
} 

__global__ void reduce_private_copies_kernel(
    float *__restrict__ result, 
    uint32_t const *__restrict__ n_centroids, 
    uint32_t const *__restrict__ n_dims) {

    uint32_t K = *n_centroids, D = *n_dims;

    int size = K * D;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < size; i += n_threads) {

        float accumulator = result[i];
        float *copy_ptr = result;

        #pragma unroll
        for(int copy = 1; copy < NUM_PRIV_COPIES; copy++) {
            copy_ptr += size;
            accumulator += copy_ptr[i];
        }

        result[i] = accumulator;
    }
}

__global__ void divide_centroids_kernel(
    float *__restrict__ centroids, 
    uint32_t const *__restrict__ counts, 
    uint32_t const *__restrict__ n_centroids, 
    uint32_t const *__restrict__ n_dims) {

    uint32_t K = *n_centroids, D = *n_dims;

    int size = K * D;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < size; i += n_threads) {
        int k = i / D;
        float div = counts[k];

        if(div > 0) {
            centroids[i] /= div;
        }
    }
}

__global__ void accumulate_cluster_members_kernel(
    float const *__restrict__ points, 
    float *__restrict__ accumulator, 
    uint32_t const *__restrict__ assignments, 
    uint32_t *__restrict__ counts, 
    uint32_t const *__restrict__ n_points,
    uint32_t const *__restrict__ n_centroids,
    uint32_t const *__restrict__ n_dims) {

    uint32_t N = *n_points, K = *n_centroids, D = *n_dims;

    float *priv_accumulator = get_privatized_pointer(accumulator, K, D);
    
    int size = N * D;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < size; i += n_threads) {
        
        int idx = i / D;
        int dim = i % D;

        float val = points[i];
        int cluster = assignments[idx];
        float *acc_val_ptr = &priv_accumulator[cluster * D + dim];

        atomicAdd(acc_val_ptr, val);
        if(dim == 0) {
            atomicAdd(&counts[cluster], 1);
        }
    }
}

__global__ void fused_assignment_accumulate_kernel(    
    float const *__restrict__ points, 
    float const *__restrict__ centroids, 
    float *__restrict__ accumulator, 
    uint32_t *__restrict__ assignments, 
    uint32_t *__restrict__ counts,
    uint32_t const *__restrict__ n_points,
    uint32_t const *__restrict__ n_centroids,
    uint32_t const *__restrict__ n_dims) {

    uint32_t K = *n_centroids, D = *n_dims;

    volatile __shared__ float shm_centroids[SHM_K][SHM_DIM];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int point_offset = tid * D;

    float dists[SHM_K];
    float min_dist = 1e30;
    uint32_t local_assignment;
    int k, d;

    // Tiled loop over K
    for(int k_block = 0; k_block < K; k_block += SHM_K) {

        // Clear distances
        #pragma unroll
        for(int i = 0; i < SHM_K; i++) {
            dists[i] = 0;
        }

        // Tiled loop over D
        for(int d_block = 0; d_block < D; d_block += SHM_DIM) {

            // Load centroids into shared memory
            for(int shm_idx = threadIdx.x; shm_idx < SHM_K * SHM_DIM; shm_idx += blockDim.x) {
                
                k = shm_idx / SHM_DIM;
                d = shm_idx % SHM_DIM;


                shm_centroids[k][d] = centroids[(k + k_block) * D + (d + d_block)];
            }

            __syncthreads();

            // Accumulate distances for this set of dimensions
            #pragma unroll
            for(k = 0; k < SHM_K; k++) {

                #pragma unroll
                for(d = 0; d < SHM_DIM; d++) {

                    float centroid_val = shm_centroids[k][d];
                    float val = centroid_val - points[point_offset + (d + d_block)];
                    dists[k] += val * val;
                }
            }

            __syncthreads();
        }

        // Reassign
        #pragma unroll
        for(k = 0; k < SHM_K; k++) {
            if(min_dist > dists[k]) {
                min_dist = dists[k];
                local_assignment = k + k_block;
            }
        }
    }

    // Write final assignment to global memory
    assignments[tid] = local_assignment;
    atomicAdd(&counts[local_assignment], 1);

    // Accumulate 
    float *priv_accumulator = get_privatized_pointer(accumulator, K, D);
    for(uint32_t d = 0; d < D; d++) {
        float val = points[point_offset + d];
        float *acc_val_ptr = &priv_accumulator[local_assignment * D + d];
        atomicAdd(acc_val_ptr, val);
    }
}