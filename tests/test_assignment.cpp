#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <kmeans_openmp.h>
#include <kernel_wrappers.h>
#include <dataset.h>

TEST_CASE("Check CUDA assigments match OpenMP") {

    uint32_t n_points = 1024, n_centroids = 16, n_dims = 16;
    Dataset dataset(n_points, n_dims, n_centroids);
    float *centroids = dataset.random_points(n_centroids);

    // Get openmp assignments
    uint32_t *omp_assignments = new uint32_t[n_points];
    compute_assignments(dataset.points, centroids, omp_assignments, n_points, n_centroids, n_dims);

    // Get CUDA assignments;
    uint32_t *cuda_assignments = new uint32_t[n_points];
    call_compute_assignments_kernel(dataset.points, centroids, cuda_assignments, n_points, n_centroids, n_dims);

    for(uint32_t i = 0; i < n_points; i++) {
        REQUIRE((omp_assignments[i] == cuda_assignments[i]));
    }
}