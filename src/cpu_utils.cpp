#include <cpu_utils.h>
#include <dataset.h>
#include <limits>

inline float dist_squared(float *arr1, uint32_t p1, float *arr2, uint32_t p2, uint32_t n_dims) {
    float dist = 0;

    #pragma omp simd reduction(+: dist) 
    for(uint32_t d = 0; d < n_dims; d++) {
        float diff = arr1[V_OFFSET(p1, d, n_dims)] - arr2[V_OFFSET(p2, d, n_dims)];
        dist += diff * diff; 
    }

    return dist;
}

void compute_assignments(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {

    #pragma omp parallel for
    for(uint32_t p = 0; p < n_points; p++) {

        uint32_t best_centroid = -1;
        float best_dist = numeric_limits<float>::max();

        for(uint32_t c = 0; c < n_centroids; c++) {
            float dist = dist_squared(points, p, centroids, c, n_dims);
            if(dist < best_dist) {
                best_dist = dist;
                best_centroid = c;
            }
        }

        assignments[p] = best_centroid;
    }
}

void recenter_centroids(float *points, float *centroids, uint32_t *assignments, uint32_t n_points, uint32_t n_centroids, uint32_t n_dims) {

    #pragma omp parallel for
    for(uint32_t c = 0; c < n_centroids; c++) {
        
        uint32_t count = 0;
        float acc[n_dims] = {0};

        for(uint32_t p = 0; p < n_points; p++) {
            if(assignments[p] == c) {
                count++;

                #pragma omp simd
                for(uint32_t d = 0; d < n_dims; d++) {
                    acc[d] += points[V_OFFSET(p, d, n_dims)];
                }
            }

            #pragma omp simd
            for(uint32_t d = 0; d < n_dims; d++) {
                centroids[V_OFFSET(c, d, n_dims)] = acc[d] / count;
            }
        }
    }
}

void CpuTimer::start() {
    start_time = chrono::steady_clock::now();
}

void CpuTimer::stop() {
    end_time = chrono::steady_clock::now();
}

float CpuTimer::elapsed_time() {
    chrono::duration<float, milli> duration = end_time - start_time;
    return duration.count();
}
