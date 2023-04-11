#include <dataset.h>
#include <random>
#include <cstring>
#include <math.h>
#include <chrono>
#include <algorithm>

float dist_squared(float *arr1, uint32_t p1, float *arr2, uint32_t p2, uint32_t n_dims) {
    float dist = 0;

    #pragma omp simd reduction(+: dist) 
    for(uint32_t d = 0; d < n_dims; d++) {
        float diff = arr1[V_OFFSET(p1, d, n_dims)] - arr2[V_OFFSET(p2, d, n_dims)];
        dist += diff * diff; 
    }

    return dist;
}

float compute_loss(float *points, uint32_t n_points, float *centroids, uint32_t n_centroids, uint32_t n_dims, uint32_t *assignments) {
    
    float loss = 0;
    #pragma omp parallel for reduction(+: loss)
    for(uint32_t p = 0; p < n_points; p++) {
        loss += dist_squared(points, p, centroids, assignments[p], n_dims);
    }

    return loss;
}

KMeansResult Dataset::kmeans_openmp(uint32_t n_centroids, uint32_t max_iters, float tol) {

    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 

    vector<float> time_per_iter;

    float prev_loss = numeric_limits<float>::max();
    
    // kmeans iteration
    for(uint32_t iter = 0; iter < max_iters; iter++) {

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();

        // assignments
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

        // recenter
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

        float curr_loss = compute_loss(points, n_points, centroids, n_centroids, n_dims, assignments);
        
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        chrono::duration<float> dur = chrono::duration_cast<chrono::milliseconds>(end - begin);
        
        time_per_iter.push_back(dur.count());

        if(iter > 0 && abs(prev_loss - curr_loss) / prev_loss < tol) {
            break;
        }

        prev_loss = curr_loss;
    }

    return KMeansResult(centroids, n_centroids, assignments, time_per_iter);
}