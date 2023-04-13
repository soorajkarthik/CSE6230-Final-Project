#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <algorithm>
#include <cpu_utils.h>
#include <kernel_wrappers.h>
#include <dataset.h>
#include <cmath>

TEST_CASE("Check CUDA fused assignment and centroid update matches OpenMP") {

    uint32_t n_points = 1024, n_centroids = 16, n_dims = 16;
    Dataset dataset(n_points, n_dims, n_centroids);

    float *omp_centroids = dataset.random_points(n_centroids);
    float *cuda_centroids = new float[n_centroids * n_dims];
    
    copy(omp_centroids, omp_centroids + n_centroids * n_dims, cuda_centroids);

    // Get assignments and recenter OpenMP
    uint32_t *omp_assignments = new uint32_t[n_points];
    compute_assignments(dataset.points, omp_centroids, omp_assignments, n_points, n_centroids, n_dims);
    recenter_centroids(dataset.points, omp_centroids, omp_assignments, n_points, n_centroids, n_dims);

    // Cuda fused assignment and recenter
    uint32_t *cuda_assignments = new uint32_t[n_points];
    call_fused_assignment_recenter_kernels(dataset.points, cuda_centroids, cuda_assignments, n_points, n_centroids, n_dims);

    for(uint32_t i = 0; i < n_points; i++) {
        REQUIRE((omp_assignments[i] == cuda_assignments[i]));
    }

    for(uint32_t i = 0; i < n_centroids * n_dims; i++) {
        REQUIRE((abs(omp_centroids[i] - cuda_centroids[i]) <= 0.001));
    }
}