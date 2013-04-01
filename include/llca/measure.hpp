#ifndef LLCA_MEASURE_HPP
#define	LLCA_MEASURE_HPP

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "llca/kernel.hpp"

namespace llca {
namespace measure {

gsl_vector_int* cluster_sizes(const gsl_vector_int &labels, int c);

/**
 * Computes the cluster balance from the cluster labels. The number of clusters
 * needs to be specified in order to permit the cluster balance measure in order
 * to factor empty clusters into the calculations.
 */
double cluster_balance(const gsl_vector_int &labels, int c);

/**
 * Computes the expected density of a clustering. The algorithm is based on
 * [1] B. Stein and S. Meyer, “On Cluster Validity and the Information Need of
 * Users,” in 3rd International Conference on Artificial Intelligence and
 * Applications, 2003, pp. 216–221. However, we only use the k nearest neighbors
 * when constructing the similarity graph.
 */
double expected_density(const gsl_matrix &data, const gsl_vector_int &labels, int c, const Kernel &ker, int k);

}
}

#endif	/* LLCA_MEASURE_HPP */
