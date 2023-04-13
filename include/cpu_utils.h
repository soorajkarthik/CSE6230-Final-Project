#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <cstdint>
#include <chrono>

using namespace std;

inline float dist_squared(float *arr1, uint32_t p1, float *arr2, uint32_t p2, uint32_t n_dims);
void compute_assignments(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims);
void recenter_centroids(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims);
float compute_loss(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims);

class CpuTimer {
    private:
        chrono::steady_clock::time_point start_time;
        chrono::steady_clock::time_point end_time;

    public:
        void start();
        void stop();
        float elapsed_time();

};

#endif