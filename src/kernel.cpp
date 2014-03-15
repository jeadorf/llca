#include "llca/kernel.hpp"

#include <cmath>

double llca::RbfKernel::operator()(const gsl_matrix &data, int i, int j) const {
    double v = 0, d = 0;
    for (size_t k = 0; k < data.size2; k++) {
        d = gsl_matrix_get(&data, i, k) - gsl_matrix_get(&data, j, k);
        v -= d * d;
    }
    return exp(v /= sigma_sq_);
}
