#include <cpu_utils.h>
#include <dataset.h>
#include <random>
#include <cstring>
#include <math.h>
#include <chrono>
#include <algorithm>

KMeansResult Dataset::kmeans_openmp(uint32_t n_centroids, uint32_t max_iters, float tol) {

    float *centroids = random_points(n_centroids);
    uint32_t *assignments = new uint32_t[n_points]; 

    vector<float> time_per_iter;

    float prev_loss = numeric_limits<float>::max();
    
    // kmeans iteration
    for(uint32_t iter = 0; iter < max_iters; iter++) {

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();

        // assignments
        compute_assignments(points, centroids, assignments, n_points, n_centroids, n_dims);

        // recenter
        recenter_centroids(points, centroids, assignments, n_points, n_centroids, n_dims);

        // compute loss
        float curr_loss = compute_loss(points, centroids, assignments, n_points, n_centroids, n_dims);
        
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