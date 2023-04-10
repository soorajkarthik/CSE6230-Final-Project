#ifndef IO_H
#define IO_H

#include <dataset.h>

void write_to_file(char *file, float *points, uint32_t n_points, uint32_t n_dims, float *centroids, uint32_t n_centroids, uint32_t *assignments);
void write_to_file(char *file, Dataset dataset, KMeansResult result);

#endif