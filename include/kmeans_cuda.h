#ifndef KMEANS_CUDA_H
#define KMEANS_CUDA_H

#include <cuda_runtime.h>
#include <cstdint>

void host_to_device_init_transfer(
    float *points, float *d_points, 
    float *centroids, float *d_centroids,
    uint32_t *assignments, uint32_t *d_assignments,
    uint32_t n_points, uint32_t *d_n_points,
    uint32_t n_centroids, uint32_t *d_n_centroids,
    uint32_t n_dims, uint32_t *d_n_dims);

void device_to_host_transfer_free(
    float *points, float *d_points, 
    float *centroids, float *d_centroids,
    uint32_t *assignments, uint32_t *d_assignments,
    uint32_t n_points, uint32_t *d_n_points,
    uint32_t n_centroids, uint32_t *d_n_centroids,
    uint32_t n_dims, uint32_t *d_n_dims);

__global__ void compute_assignments_kernel(
    float *__restrict__ points,
    float *__restrict__ centroids,
    uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ n_points,
    uint32_t *__restrict__ n_centroids,
    uint32_t *__restrict__ n_dims);

#endif