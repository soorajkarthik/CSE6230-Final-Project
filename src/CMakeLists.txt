add_library(kmeans "")
target_sources(kmeans PRIVATE dataset.cpp io.cpp kmeans_openmp.cpp kmeans_cuda.cu kernel_wrappers.cu device_utils.cu cpu_utils.cpp kernels.cu)
target_include_directories(kmeans PUBLIC ../include)
target_link_libraries(kmeans PUBLIC CUDA::cudart)