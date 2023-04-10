#include <cstdio>
#include <dataset.h>
#include <io.h>

int main(int argc, char *argv[]) {
    Dataset d(10, 2, 10);
    KMeansResult res = d.kmeans_openmp(10, 1000, 0);

    for(auto l: res.time_per_iter) {
        printf("%.5f\n", l);
    }

    write_to_file(argv[1], d, res);
}