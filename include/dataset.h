#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cstdint>
#define V_OFFSET(p, d, n_dims) ((p) * (n_dims) + (d))
#define P_OFFSET(p, n_dims) ((p) * (n_dims))

using namespace std;


class KMeansResult {
    public:
        vector<float> centroids;
        vector<uint32_t> assignments;
        vector<float> loss_per_iter;
        vector<float> time_per_iter;

        KMeansResult(vector<float> centroids, vector<uint32_t> assignments, vector<float> loss_per_iter, vector<float> time_per_iter);
};


class Dataset {
    uint32_t n_points;
    uint32_t n_dims;
    vector<float> values;

    public:
        Dataset();
        Dataset(uint32_t n_points, uint32_t n_dims);

        void randinit();
        void print();

        vector<float> random_points(uint32_t num);
        KMeansResult kmeans_openmp(uint32_t n_centroids, uint32_t max_iters, float tol);
        KMeansResult kmeans_cuda(uint32_t n_centroids, uint32_t max_iters, float tol);

};

#endif