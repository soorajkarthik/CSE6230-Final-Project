#include <dataset.h>
#include <random>
#include <cstdio>
#include <algorithm>

Dataset::Dataset() : n_points{0}, n_dims{0} {}

Dataset::Dataset(uint32_t n_points, uint32_t n_dims) : n_points{n_points}, n_dims{n_dims} {
    values = new float[n_points * n_dims];
    randinit();
}

void Dataset::randinit() {
    random_device r;
    mt19937 gen(r());
    uniform_real_distribution<float> distr(1.0, 5.0);

    for(uint32_t i = 0; i < n_points * n_dims; i++)
        values[i] = distr(gen);
}

void Dataset::print() {
    uint32_t i, j;
    for(i = 0; i < n_points; i++) {
        for(j = 0; j < n_dims; j++) {
            printf("%.2f ", values[V_OFFSET(i, j, n_dims)]);
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
            values + P_OFFSET(idx, n_dims), 
            values + P_OFFSET(idx + 1, n_dims), 
            res + P_OFFSET(i, n_dims)
        );
    }

    return res;
}

KMeansResult::KMeansResult(float *centroids, uint32_t *assignments, vector<float> loss_per_iter, vector<float> time_per_iter) 
    : centroids{centroids}, assignments{assignments}, loss_per_iter{loss_per_iter}, time_per_iter{time_per_iter} {}