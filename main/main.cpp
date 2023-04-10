#include <cstdio>
#include <dataset.h>
#include <io.h>

int main(int argc, char *argv[]) {
    Dataset d(1000, 2, 5);
    KMeansResult res = d.kmeans_openmp(5, 10, 1e-7);
    write_to_file(argv[1], d, res);
}