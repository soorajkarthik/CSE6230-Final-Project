#include <cstdio>
#include <dataset.h>

int main() {
    Dataset d(100000, 10);
    KMeansResult res = d.kmeans_openmp(5, 1, 0.0000001);

    for(auto l: res.time_per_iter) {
        printf("%.5f\n", l);
    }
}