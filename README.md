# CUDA Acceleration for K-Means Clustering

In this project, we explored various ways to optimize the k-means clustering algorithm for execution on NVIDIA GPUs. 

## Prepare the environment on PACE

```
module load gcc cuda cmake
```

 
## Checkout the source code

Clone the repository as follows

```
git clone --recurse-submodules https://github.com/soorajkarthik/CSE6230-Final-Project
```

  

## To Compile

  

Set the build version in `build.config` to Debug or Release then run the `build.sh`: 
```
bash build.sh
```

  

## To Run Tests

  

See the executables under the `build/tests` directory. There should be three: `test_assignment`, `test_update`, and `test_fused`.

  

## To Run Code

Use the main executable found under the `build/main` directory. It should just be called `main`. Here is the format for running the executable.
```
./build/main/main -a <alg> -i <num iters> -n <num points> -k <num centroids> -d <num dims> 
```
Use the `-s` flag to save the k-means output to a text file and use the `-h` flag for more details on each option that the executable accepts.
