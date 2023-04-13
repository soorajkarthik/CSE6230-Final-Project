#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include <cstdint>

void host_to_device_init_transfer(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    float **d_accumulator, uint32_t **d_counts,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims);

void device_to_host_transfer_free(
    float *points, float **d_points, 
    float *centroids, float **d_centroids,
    uint32_t *assignments, uint32_t **d_assignments,
    float **d_accumulator, uint32_t **d_counts,
    uint32_t n_points, uint32_t **d_n_points,
    uint32_t n_centroids, uint32_t **d_n_centroids,
    uint32_t n_dims, uint32_t **d_n_dims);
    

class CudaTimer {
    private:
        cudaEvent_t start_event;
        cudaEvent_t stop_event;

    public:
        void start();
        void stop();
        float elapsed_time();

};


#endif