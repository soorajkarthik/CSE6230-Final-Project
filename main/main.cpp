#include <cstdio>
#include <dataset.h>
#include <io.h>
#include <iostream>

int main(int argc, char *argv[]) {

    uint32_t n_iters = 100, n_points = 4000, n_dims = 2, n_centroids = 16;
    bool cuda_fused_kernel = false;

    Dataset d(n_points, n_dims, n_centroids);

    KMeansResult res_openmp = d.kmeans_openmp(n_centroids, n_iters);
    write_to_file(argv[1], d, res_openmp);

    KMeansResult res_cuda = d.kmeans_cuda(n_centroids, n_iters, cuda_fused_kernel);
    write_to_file(argv[2], d, res_cuda);

    float cpu_time = 0, gpu_time = 0;
    for(uint32_t i = 0; i < n_iters; i++) {
        cpu_time += res_openmp.time_per_iter[i];
        gpu_time += res_cuda.time_per_iter[i];
    }

    cpu_time /= n_iters;
    gpu_time /= n_iters;

    printf("Avg CPU Time: %f\nAvg GPU Time: %f\n", cpu_time, gpu_time);
}