#include <cstdio>
#include <dataset.h>
#include <io.h>

int main() {
    Dataset d(100000, 10);
    KMeansResult res = d.kmeans_openmp(5, 100, 0);

    for(auto l: res.time_per_iter) {
        printf("%.5f\n", l);
    }

    write_to_file("out.txt", d, res);
}