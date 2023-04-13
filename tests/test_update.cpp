#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <algorithm>
#include <cpu_utils.h>
#include <kernel_wrappers.h>
#include <dataset.h>
#include <cmath>

TEST_CASE("Check CUDA centroid update matches OpenMP") {

    uint32_t n_points = 1024, n_centroids = 16, n_dims = 16;
    Dataset dataset(n_points, n_dims, n_centroids);
    float *centroids = dataset.random_points(n_centroids);
    
    float *centroids_gpu = new float[n_centroids * n_dims];
    copy(centroids, centroids + n_centroids * n_dims, centroids_gpu);

    // Get assignments
    uint32_t *assignments = new uint32_t[n_points];
    compute_assignments(dataset.points, centroids, assignments, n_points, n_centroids, n_dims);

    // OpenMP recenter
    recenter_centroids(dataset.points, centroids, assignments, n_points, n_centroids, n_dims);

    // Cuda recenter
    call_recenter_centroids_kernels(dataset.points, centroids_gpu, assignments, n_points, n_centroids, n_dims);

    for(uint32_t i = 0; i < n_centroids * n_dims; i++) {
        REQUIRE((abs(centroids[i] - centroids_gpu[i]) <= 0.001));
    }
}