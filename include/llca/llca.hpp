#ifndef LLCA_LLCA_HPP
#define	LLCA_LLCA_HPP

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace llca {

struct Workspace {

    gsl_matrix *A;
    
    gsl_matrix *T;
    
    gsl_matrix *scores;
    
    int n;
    
    int c;

    Workspace(int n, int c);

    ~Workspace();
    
};

/**
 * Clusters the data using LLCA.
 *
 * [1] M. Wu and B. Schölkopf, “A Local Learning Approach for Clustering,” in
 *     Advances in Neural Information Processing Systems, 2006.
 *
 * @param data          data, where rows correspond to data points
 * @param c             number of clusters
 * @param sigma         kernel bandwidth parameter
 * @param k             number of nearest neighbors to consider
 * @param lambda        regularization parameter
 * @param workspace     workspace, can be used for debugging purposes.
 *                      The workspace needs to be appropriately initialized.
 *
 * @return  a vector with the clustering, caller is responsible for freeing
 */
gsl_vector_int* cluster(const gsl_matrix &data,
                            int c,
                            double sigma,
                            int k,
                            double lambda,
                            Workspace &workspace);

/**
 * Computes cluster labels for a subsample of the data using LLCA and uses a
 * SVM-based out-of-sample extension to compute the cluster labels for the
 * complete data. The percentage of the data points used in LLCA is controlled
 * by parameter svm_quota.
 */
gsl_vector_int* cluster_with_oosext(const gsl_matrix &data,
                            int c,
                            double sigma,
                            int k,
                            double lambda,
                            double svm_quota,
                            double svm_gamma,
                            double svm_cost);

/**
 * Same as cluster_wiht_oosext, however the workspace is returned via the last
 * parameter; it is the caller's responsibility to delete the workspace.
 */
gsl_vector_int* cluster_with_oosext(const gsl_matrix &data,
                            int c,
                            double sigma,
                            int k,
                            double lambda,
                            double svm_quota,
                            double svm_gamma,
                            double svm_cost,
                            Workspace *&workspace);

}

#endif	/* LLCA_LLCA_HPP */
