#ifndef KMEANS_CUDA_H
#define KMEANS_CUDA_H

#include <cuda_runtime.h>
#include <cstdint>

__global__ void compute_assignments_kernel(
    float *__restrict__ points,
    float *__restrict__ centroids,
    uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ n_points,
    uint32_t *__restrict__ n_centroids,
    uint32_t *__restrict__ n_dims);

#endif