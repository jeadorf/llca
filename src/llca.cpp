#include "llca/llca.hpp"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <set>
#include <stdexcept>
#include <vector>

#include "llca/kernel.hpp"
#include "llca/kmeans.hpp"
#include "llca/knn.hpp"
#include "llca/svm.hpp"

 llca::Workspace::Workspace(int n, int c) : n(n), c(c) {
     A = gsl_matrix_alloc(n, n);
     T = gsl_matrix_alloc(n, n);
     scores = gsl_matrix_alloc(n, c);
 }
    
llca::Workspace::~Workspace() {
    gsl_matrix_free(A);
    gsl_matrix_free(T);
    gsl_matrix_free(scores);
}

gsl_matrix* local_krr_matrix(const int index,
                                const gsl_matrix &data,
                                const llca::NearestNeighbors &nn,
                                const llca::Kernel &ker,
                                const double lambda) {
    const int k = nn.get_k();
    gsl_matrix *M = gsl_matrix_calloc(k, k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            gsl_matrix_set(M, i, j, ker(data, nn(index, i), nn(index, j)));
        }
    }
    for (int i = 0; i < k; i++) {
        gsl_matrix_set(M, i, i, lambda + gsl_matrix_get(M, i, i));
    }
    return M;

}

gsl_vector* local_krr_rhs(const int index,
                                const gsl_matrix &data,
                                const llca::NearestNeighbors &nn,
                                const llca::Kernel &ker) {
    const int k = nn.get_k();
    gsl_vector *b = gsl_vector_alloc(k);
    for (int j = 0; j < k; j++) {
        gsl_vector_set(b, j, ker(data, index, nn(index, j)));
    }
    return b;
}

gsl_vector* solve_linear_system(gsl_matrix *M, gsl_vector *b) {
    gsl_vector *x = gsl_vector_alloc(M->size1);
    gsl_permutation * p = gsl_permutation_alloc(M->size1);
    int s;
    
    gsl_linalg_LU_decomp(M, p, &s);
    gsl_linalg_LU_solve(M, p, b, x);

    gsl_permutation_free(p);
       
    return x;
}

void compute_matrix_A(const gsl_matrix &data,
                      const double sigma,
                      const int k,
                      const double lambda,
                      llca::Workspace &ws) {
    gsl_matrix_set_zero(ws.A);
    
    llca::RbfKernel ker(sigma);
    llca::BruteForceNearestNeighbors nn(&data, k);
    
    #pragma omp parallel for
    for (int i = 0; i < ws.n; i++) {
        gsl_matrix *M_i = local_krr_matrix(i, data, nn, ker, lambda);
        gsl_vector *k_i = local_krr_rhs(i, data, nn, ker);
        gsl_vector *alpha = solve_linear_system(M_i, k_i);
        
        for (int j = 0; j < nn.get_k(); j++) {
            gsl_matrix_set(ws.A, i, nn(i, j), gsl_vector_get(alpha, j));
        }
        
        gsl_vector_free(alpha);
        gsl_vector_free(k_i);
        gsl_matrix_free(M_i);
    }
}

void compute_matrix_T(llca::Workspace &ws) {
    const int n = ws.n;
    gsl_matrix *tmp = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(tmp, ws.A);
    gsl_matrix_scale(tmp, -1);
    for (int i = 0; i < n; i++) {
        gsl_matrix_set(tmp, i, i, 1 + gsl_matrix_get(tmp, i, i));
    }
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, tmp, tmp, 0.0, ws.T);
    gsl_matrix_free(tmp);
}

void compute_scores(llca::Workspace &ws) {
    const int n = ws.n, c = ws.c;
    gsl_vector *eval = gsl_vector_alloc(n);
    gsl_matrix *evec = gsl_matrix_alloc(n, n);
    gsl_eigen_symmv_workspace * eigws = gsl_eigen_symmv_alloc(n);
    gsl_matrix *T_copy = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(T_copy, ws.T);
    gsl_eigen_symmv(T_copy, eval, evec, eigws);
    gsl_matrix_free(T_copy);
    gsl_eigen_symmv_free(eigws);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
    gsl_vector_free(eval);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            gsl_matrix_set(ws.scores, i, j, gsl_matrix_get(evec, i, j));
        }
    }
    gsl_matrix_free(evec);
}

gsl_vector_int* compute_labels(llca::Workspace &ws) {
    return llca::kmeans(*ws.scores, ws.c);
}

gsl_vector_int* llca::cluster(const gsl_matrix &data,
                                    const int c,
                                    const double sigma,
                                    const int k,
                                    const double lambda,
                                    Workspace &workspace) {
    if (c <= 0) throw std::runtime_error("c must be positive.");
    if (sigma <= 0) throw std::runtime_error("sigma must be positive.");
    if (k <= 0) throw std::runtime_error("k must be positive.");
    if (lambda <= 0) throw std::runtime_error("lambda must be positive.");
    
    compute_matrix_A(data, sigma, k, lambda, workspace);
    compute_matrix_T(workspace);
    compute_scores(workspace);
    return compute_labels(workspace);
}

gsl_matrix* random_sample(const gsl_matrix &data, const int m) {
    const int n = data.size1;
    const int d = data.size2;
    gsl_matrix* sample = gsl_matrix_alloc(m, d);
    std::vector<int> p(n);
    for (int i = 0; i < n; i++) {
        p[i] = i;
    }
    std::random_shuffle(p.begin(), p.end());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            gsl_matrix_set(sample, i, j, gsl_matrix_get(&data, p[i], j));
        }
    }
    return sample;
}

gsl_vector_int* llca::cluster_with_oosext(const gsl_matrix &data,
                            int c,
                            double sigma,
                            int k,
                            double lambda,
                            double svm_quota,
                            double svm_gamma,
                            double svm_cost) {
    llca::Workspace *ws;
    gsl_vector_int *labels = llca::cluster_with_oosext(data, c, sigma, k, lambda,
                                            svm_quota, svm_gamma, svm_cost, ws);
    delete ws;
    return labels;
}

gsl_vector_int* llca::cluster_with_oosext(const gsl_matrix &data,
                            int c,
                            double sigma,
                            int k,
                            double lambda,
                            double svm_quota,
                            double svm_gamma,
                            double svm_cost,
                            llca::Workspace *&workspace) {
    const int n = data.size1;
    const int m = n * svm_quota;
    gsl_matrix* train = random_sample(data, m);
    workspace = new llca::Workspace(train->size1, c);
    gsl_vector_int* train_labels = cluster(*train, c, sigma, k, lambda, *workspace);

    SVM svm(svm_gamma, svm_cost);
    svm.train(*train, *train_labels);
    gsl_vector_int *labels = svm.predict(data);

    gsl_matrix_free(train);
    gsl_vector_int_free(train_labels);
    return labels;
}
