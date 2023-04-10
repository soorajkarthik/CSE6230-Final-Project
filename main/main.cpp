#include <cstdio>
#include <dataset.h>
#include <io.h>

int main(int argc, char *argv[]) {
    Dataset d(10000, 2, 10);
    KMeansResult res = d.kmeans_openmp(10, 1000000, 1e-7);
    write_to_file(argv[1], d, res);
}