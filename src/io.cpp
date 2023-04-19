#include <io.h>
#include <fstream>

void write_to_file(string file, float *points, uint32_t n_points, uint32_t n_dims, float *centroids, uint32_t n_centroids, uint32_t *assignments) {
    ofstream outfile(file, fstream::out);

    outfile << n_points << " " << n_dims << endl;
    for(uint32_t p = 0; p < n_points; p++) {
        for(uint32_t d = 0; d < n_dims; d++) {
            outfile << points[p * n_dims + d] << " ";
        }
        outfile << endl;
    }

    outfile << n_centroids << " " << n_dims << endl;
    for(uint32_t c = 0; c < n_centroids; c++) {
        for(uint32_t d = 0; d < n_dims; d++) {
            outfile << centroids[c * n_dims + d] << " ";
        }
        outfile << endl;
    }

    for(uint32_t p = 0; p < n_points; p++) {
        outfile << assignments[p] << " ";
    }

    outfile.close();
}

void write_to_file(string file, Dataset dataset, KMeansResult result) {
    write_to_file(file, dataset.points, dataset.n_points, dataset.n_dims, result.centroids, result.n_centroids, result.assignments);
}