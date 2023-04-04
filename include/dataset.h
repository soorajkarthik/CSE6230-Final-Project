#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cstdint>

class Dataset {
    std::uint64_t n_points;
    std::uint64_t n_dims;
    std::vector<double> values;

    public:
        Dataset();
        Dataset(std::uint64_t n_points, std::uint64_t n_dims);
        
        Dataset(Dataset&&) = default;
        Dataset(const Dataset&) = default;
        
        Dataset& operator=(const Dataset&) = default;
        Dataset& operator=(Dataset&&) = default;
        
        ~Dataset() = default;

        void randinit();
        void print();

        std::vector<double> kmeans_openmp(std::uint64_t n_centroids);
        std::vector<double> kmeans_cuda(std::uint64_t n_centroids);

};

#endif