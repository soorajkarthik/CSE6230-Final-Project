#ifndef KERNEL_WRAPPERS_H
#define KERNEL_WRAPPERS_H

#include <cstdint>

void call_compute_assignments_kernel(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims);

#endif