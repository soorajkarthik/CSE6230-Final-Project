#include <cstdio>
#include <dataset.h>
#include <io.h>
#include <iostream>

int main(int argc, char *argv[]) {
    Dataset d(4000, 2, 16);

    KMeansResult res_openmp = d.kmeans_openmp(16, 1000000, 1e-7);
    write_to_file(argv[1], d, res_openmp);

    KMeansResult res_cuda = d.kmeans_cuda(16, 1000000, 1e-7);
    write_to_file(argv[2], d, res_cuda);

    if (res_cuda != res_openmp){
        std::cerr << "Error: There's a mismatch between openMP and Cuda results";
        return -1; 
    }
}