#ifndef LLCA_KMEANS_HPP
#define	LLCA_KMEANS_HPP

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace llca {

gsl_vector_int* kmeans(const gsl_matrix &data, int c);

}

#endif /* LLCA_KMEANS_HPP */
