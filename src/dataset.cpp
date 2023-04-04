#include <dataset.h>
#include <random>
#include <cstdio>

Dataset::Dataset() : n_points{0}, n_dims{0} {}

Dataset::Dataset(std::uint64_t n_points, std::uint64_t n_dims) : n_points{n_points}, n_dims{n_dims} {
    values.resize(n_points * n_dims);
    randinit();
}

void Dataset::randinit() {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_real_distribution<> distr(1.0, 5.0);

    for(std::uint64_t i = 0; i < n_points * n_dims; i++)
        values[i] = distr(gen);
}

void Dataset::print() {
    std::uint64_t i, j;
    for(i = 0; i < n_points; i++) {
        for(j = 0; j < n_dims; j++) {
            printf("%.2f ", values[j * n_points + i]);
        }

        printf("\n");
    }
}