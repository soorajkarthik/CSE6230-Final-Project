#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cstdint>

#define V_OFFSET(p, d, n_dims) ((p) * (n_dims) + (d))
#define P_OFFSET(p, n_dims) ((p) * (n_dims))

using namespace std;

class KMeansResult {
    public:
        float* centroids;
        uint32_t n_centroids;
        uint32_t* assignments;
        vector<float> time_per_iter;

        KMeansResult() = default;
        KMeansResult(float *centroids, uint32_t n_centroids, uint32_t *assignments, vector<float> time_per_iter);

        bool operator!=(const KMeansResult& rhs);
};

class Dataset {

    public:
        uint32_t n_points;
        uint32_t n_dims;
        float *points;
        
        Dataset();
        Dataset(uint32_t n_points, uint32_t n_dims, uint32_t n_clusters);

        void randinit(uint32_t n_clusters);
        void print();

        float* random_points(uint32_t num);
        float* get_tranposed_points();
        
        KMeansResult kmeans_openmp(uint32_t n_centroids, uint32_t max_iters);
        KMeansResult kmeans_cuda(uint32_t n_centroids, uint32_t max_iters, bool fused_kernel);

};

#endif