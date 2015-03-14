## About llca ##

**`llca` clusters data**. It is a C++ implementation of M. Wu and B. Schölkopf, “[A Local Learning Approach for Clustering](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.69.2181)”, in Advances in Neural Information Processing Systems, 2006. The paper version has been extended by a SVM-based out-of-sample extension in order to make it possible to cluster large datasets.

The code has been inspired by existing code at the [Chair of Scientific Computing](http://www5.in.tum.de) at Technische Universität München. This sofware has been written in the course of a research project at Technische Universität München and Royal Institute of Technology.

## Building llca ##

`llca` uses CMake as build system. It depends on [kmlocal](http://www.cs.umd.edu/~mount/Projects/KMeans) and [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/). Both dependencies have to be downloaded and extracted separately to your preferred location. The code has been tested against kmlocal-1.7.2 and libsvm-3.17.

```
git clone https://code.google.com/p/llca/
cd llca
mkdir build
cd build
cmake -D KMLOCAL_SOURCE_DIR=/path/to/kmlocal-1.7.2 -D LIBSVM_SOURCE_DIR=/path/to/libsvm-3.17 ..
make
```

## Testing llca ##

There are a couple of unit tests. Go to the build directory and run the tests as follows

```
    make test
```