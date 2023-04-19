#ifndef IO_H
#define IO_H

#include <dataset.h>
#include <string>

void write_to_file(string file, float *points, uint32_t n_points, uint32_t n_dims, float *centroids, uint32_t n_centroids, uint32_t *assignments);
void write_to_file(string file, Dataset dataset, KMeansResult result);

#endif