#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

__global__ void compute_assignments_kernel(
    float *__restrict__ points,
    float *__restrict__ centroids,
    uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ n_points,
    uint32_t *__restrict__ n_centroids,
    uint32_t *__restrict__ n_dims);

__global__ void reduce_private_copies_kernel(
    float *result, 
    uint32_t *n_centroids, 
    uint32_t *n_dims);

__global__ void divide_centroids_kernel(
    float *centroids, 
    uint32_t *counts, 
    uint32_t *n_centroids, 
    uint32_t *n_dims);

__global__ void accumulate_cluster_members_kernel(
    float *points, 
    float *accumulator, 
    uint32_t *assignments, 
    uint32_t *counts, 
    uint32_t *n_points,
    uint32_t *n_centroids,
    uint32_t *n_dims);

__global__ void fused_assignment_accumulate_kernel(    
    float *points, 
    float *centroids, 
    float *accumulator, 
    uint32_t *assignments, 
    uint32_t *counts,
    uint32_t *n_points,
    uint32_t *n_centroids,
    uint32_t *n_dims);

#endif