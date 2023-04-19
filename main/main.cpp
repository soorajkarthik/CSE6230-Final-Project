#include <dataset.h>
#include <io.h>
#include <iostream>
#include <tclap/CmdLine.h>

int main(int argc, char** argv) {

    TCLAP::CmdLine cmd("Accelerated KMeans Clustering", (char)32, "0.1");

    vector<string> allowed_algs = {"openmp", "cudamulti", "cudafused"};
    TCLAP::ValuesConstraint<string> constraint_algs(allowed_algs);

    TCLAP::ValueArg<string> arg_algs("a", "alg", "algorithm to use", false, "openmp", &constraint_algs, cmd);
    TCLAP::ValueArg<uint32_t> arg_iters("i", "iters", "number of iterations to run", false, 100, "int", cmd);
    TCLAP::ValueArg<uint32_t> arg_points("n", "points", "number of points to input", false, 100000, "int", cmd);
    TCLAP::ValueArg<uint32_t> arg_centroids("k", "centroids", "number of centroids to use", false, 32, "int", cmd);
    TCLAP::ValueArg<uint32_t> arg_dims("d", "dims", "dimensionality of input data", false, 32, "int", cmd);
    TCLAP::SwitchArg arg_write_output("s", "save", "whether or not to save output to file", cmd, false);

    cmd.parse(argc, argv);

    string alg = arg_algs.getValue();
    uint32_t n_iters = arg_iters.getValue();
    uint32_t n_points = arg_points.getValue();
    uint32_t n_dims = arg_dims.getValue();
    uint32_t n_centroids = arg_dims.getValue();
    bool write_output = arg_write_output.getValue();

    cout << "Initializing dataset..." << endl;

    Dataset dataset(n_points, n_dims, n_centroids);
    KMeansResult result;

    cout << "Running " << n_iters << " iterations of \"" << alg << "\" algorithm..." << endl;

    if(alg == "openmp") {
        result = dataset.kmeans_openmp(n_centroids, n_iters);
    } else if (alg == "cudamulti") {
        result = dataset.kmeans_cuda(n_centroids, n_iters, false);
    } else if (alg == "cudafused") {
        result = dataset.kmeans_cuda(n_centroids, n_iters, true);
    }

    if(write_output) {
        cout << "Writing results to kmeans_out.txt..." << endl;
        write_to_file("kmeans_out.txt", dataset, result);
    }

    float avg_time = 0;
    for(uint32_t i = 0; i < n_iters; i++) {
        avg_time += result.time_per_iter[i];
    }

    avg_time /= n_iters;

    cout << "Average time per iteration: " << avg_time << endl;
}