add_library(kmeans "")
target_sources(kmeans PRIVATE dataset.cpp kmeans_openmp.cpp)
target_include_directories(kmeans PUBLIC ../include)
# target_link_libraries(kmeans PUBLIC BLAS::BLAS CUDA::cudart CUDA::cublas)