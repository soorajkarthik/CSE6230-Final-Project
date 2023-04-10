#include <dataset.h>
#include <random>
#include <cstdio>
#include <algorithm>
#include <iostream>

Dataset::Dataset() : n_points{0}, n_dims{0} {}

Dataset::Dataset(uint32_t n_points, uint32_t n_dims, uint32_t n_clusters) : n_points{n_points}, n_dims{n_dims} {
    points = new float[n_points * n_dims];
    randinit(n_clusters);
}

void Dataset::randinit(uint32_t n_clusters) {

    #pragma omp parallel for
    for(uint32_t c = 0; c < n_clusters; c++) {
        
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> gauss_dist(0, 1);
        uniform_int_distribution<int> int_dist(-100, 100);

        float center[n_dims];
        for(uint32_t d = 0; d < n_dims; d++) {
            center[d] = int_dist(gen);
        }

        for(uint32_t p = c; p < n_points; p += n_clusters) {
            for(uint32_t d = 0; d < n_dims; d++) {
                points[V_OFFSET(p, d, n_dims)] = gauss_dist(gen) + center[d];
            }
        }
    }
}

void Dataset::print() {
    uint32_t i, j;
    for(i = 0; i < n_points; i++) {
        for(j = 0; j < n_dims; j++) {
            printf("%.2f ", points[V_OFFSET(i, j, n_dims)]);
        }
        
        printf("\n");
    }
}

float* Dataset::random_points(uint32_t num) {
    vector<int> sample(n_points);
    iota(sample.begin(), sample.end(), 0);
    shuffle(sample.begin(), sample.end(), default_random_engine());
    
    float *res = new float[num * n_dims];
    for(uint32_t i = 0; i < num; i++) {
        int idx = sample[i];
        copy(
            points + P_OFFSET(idx, n_dims), 
            points + P_OFFSET(idx + 1, n_dims), 
            res + P_OFFSET(i, n_dims)
        );
    }

    return res;
}

KMeansResult::KMeansResult(float *centroids, uint32_t n_centroids, uint32_t *assignments, vector<float> loss_per_iter, vector<float> time_per_iter) {
    this->centroids = centroids;
    this->n_centroids = n_centroids;
    this->assignments = assignments;
    this->loss_per_iter = loss_per_iter;
    this->time_per_iter = time_per_iter;
}