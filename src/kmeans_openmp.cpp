#include <dataset.h>
#include <random>
#include <cstring>
#include <math.h>
#include <chrono>
#include <algorithm>

float dist_squared(vector<float> arr1, uint32_t p1, vector<float> arr2, uint32_t p2, uint32_t n_dims) {
    float dist = 0;

    #pragma omp simd reduction(+: dist) 
    for(uint32_t d = 0; d < n_dims; d++) {
        float diff = arr1[V_OFFSET(p1, d, n_dims)] - arr2[V_OFFSET(p2, d, n_dims)];
        dist += diff * diff; 
    }

    return dist;
}

float compute_loss(vector<float> points, uint32_t n_points, vector<float> centroids, uint32_t n_centroids, uint32_t n_dims, vector<uint32_t> assignments) {
    
    float loss = 0;
    #pragma omp parallel for reduction(+: loss)
    for(uint32_t p = 0; p < n_points; p++) {
        loss += dist_squared(points, p, centroids, assignments[p], n_dims);
    }

    return loss;
}

inline void update_accumulator(vector<float> &acc, uint32_t centroid, vector<float> values, uint32_t p, uint32_t n_dims) {
    #pragma omp simd
    for(uint32_t d = 0; d < n_dims; d++) {
        acc[V_OFFSET(centroid, d, n_dims)] += values[V_OFFSET(p, d, n_dims)];
    }
} 

#pragma omp declare reduction(vec_float_plus : vector<float> : \
                                transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<float>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_int_plus : vector<uint32_t> : \
                                transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<uint32_t>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

KMeansResult Dataset::kmeans_openmp(uint32_t n_centroids, uint32_t max_iters, float tol) {

    vector<float> centroids = random_points(n_centroids);
    vector<uint32_t> assignments(n_points);
    vector<float> loss_per_iter;
    vector<float> time_per_iter;

    float prev_loss = numeric_limits<float>::max();
    
    // kmeans iteration
    for(uint32_t iter = 0; iter < max_iters; iter++) {

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();

        vector<float> accumulator(centroids.size(), 0.0);
        vector<uint32_t> counts(n_centroids, 0);

        // assignments
        #pragma omp parallel for reduction(vec_float_plus: accumulator) reduction(vec_int_plus: counts)
        for(uint32_t p = 0; p < n_points; p++) {
            uint32_t best_centroid = -1;
            float best_dist = numeric_limits<float>::max();

            for(uint32_t c = 0; c < n_centroids; c++) {
                float dist = dist_squared(values, p, centroids, c, n_dims);
                if(dist < best_dist) {
                    best_dist = dist;
                    best_centroid = c;
                }
            }

            assignments[p] = best_centroid;

            counts[best_centroid]++;
            update_accumulator(accumulator, best_centroid, values, p, n_dims);
        }

        // recenter
        #pragma omp simd
        for(uint32_t i = 0; i < centroids.size(); i++) {
            centroids[i] = accumulator[i] / counts[i / n_dims];
        }

        float curr_loss = compute_loss(values, n_points, centroids, n_centroids, n_dims, assignments);
        
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        chrono::duration<float> dur = chrono::duration_cast<chrono::milliseconds>(end - begin);
        
        loss_per_iter.push_back(curr_loss);
        time_per_iter.push_back(dur.count());

        if(iter > 0 && abs(prev_loss - curr_loss) / prev_loss < tol) {
            break;
        }

        prev_loss = curr_loss;
    }

    return KMeansResult(centroids, assignments, loss_per_iter, time_per_iter);
}