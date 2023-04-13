#include <device_utils.h>
#include <kernels.h>
#include <consts.h>

void host_to_device_init_transfer(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    float **d_accumulator, uint32_t **d_sizes,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims) {

    cudaMalloc(d_points,      n_points * n_dims * sizeof(float));
    cudaMalloc(d_centroids,   n_centroids * n_dims * sizeof(float));
    cudaMalloc(d_assignments, n_points * sizeof(uint32_t));
    cudaMalloc(d_accumulator, n_centroids * n_dims * NUM_PRIV_COPIES * sizeof(float));
    cudaMalloc(d_sizes,       n_centroids * sizeof(uint32_t));
    cudaMalloc(d_n_points,    sizeof(uint32_t));
    cudaMalloc(d_n_centroids, sizeof(uint32_t));
    cudaMalloc(d_n_dims,      sizeof(uint32_t));

    cudaMemcpy(*d_points,      points,       n_points * n_dims * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(*d_centroids,   centroids,    n_centroids * n_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_points,    &n_points,    sizeof(uint32_t),                     cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_centroids, &n_centroids, sizeof(uint32_t),                     cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_dims,      &n_dims, 	 sizeof(uint32_t),                     cudaMemcpyHostToDevice);

    cudaMemset(d_accumulator, 0, n_centroids * n_dims * NUM_PRIV_COPIES * sizeof(float));
}

void device_to_host_transfer_free(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    float **d_accumulator, uint32_t **d_sizes,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims) {

    cudaMemcpy(centroids,   *d_centroids, n_centroids * n_dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignments, *d_assignments, n_points * sizeof(uint32_t),        cudaMemcpyDeviceToHost);

    cudaFree(&d_points);
    cudaFree(&d_centroids);
    cudaFree(&d_assignments);
    cudaFree(&d_accumulator);
    cudaFree(&d_sizes);
    cudaFree(&d_n_points);
    cudaFree(&d_n_centroids);
    cudaFree(&d_n_dims);
}
