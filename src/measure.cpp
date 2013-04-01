#include "llca/measure.hpp"

#include <cassert>
#include <cmath>
#include <set>

#include "llca/knn.hpp"

gsl_vector_int* llca::measure::cluster_sizes(const gsl_vector_int &labels, int c) {
    gsl_vector_int *sz = gsl_vector_int_calloc(c);
    for (size_t i = 0; i < labels.size; i++) {
        int l = gsl_vector_int_get(&labels, i);
        assert(l >= 1 && l<= c);
        (*gsl_vector_int_ptr(sz, l - 1))++;
    }
    return sz;
}

double llca::measure::cluster_balance(const gsl_vector_int& labels, const int c) {
    int minsz, maxsz;
    gsl_vector_int *sz = cluster_sizes(labels, c);
    gsl_vector_int_minmax(sz, &minsz, &maxsz);
    gsl_vector_int_free(sz);
    return 1.0 * minsz / maxsz;
}

double graph_weight(const gsl_matrix& data, const gsl_vector_int& labels, const llca::NearestNeighbors &nn, const llca::Kernel &ker, int i) {
    typedef std::pair<int, int> Edge;
    std::set<Edge> edges;
    double weight = 0;
    for (size_t j = 0; j < labels.size; j++) {
        if (gsl_vector_int_get(&labels, j) == i) {
            for (int l = 0; l < nn.get_k(); l++) {
                if (gsl_vector_int_get(&labels, nn(j, l)) == i) {
                    Edge e_out(j, nn(j, l));
                    Edge e_in(nn(j, l), j);
                    if (edges.count(e_in) == 0) {
                        weight += ker(data, e_out.first, e_out.second);
                        edges.insert(e_out);
                    }
                }
            }
        }
    }
    return weight;
}

double llca::measure::expected_density(const gsl_matrix& data, const gsl_vector_int& labels, int c, const Kernel &ker, int k) {
    double rho = 0.0;
    gsl_vector_int *sz = cluster_sizes(labels, c);
    if (gsl_vector_int_min(sz) >= 1) {
        llca::BruteForceNearestNeighbors nn(&data, k);
        gsl_vector_int *zeros = gsl_vector_int_calloc(labels.size);
        double w = graph_weight(data, *zeros, nn, ker, 0);
        gsl_vector_int_free(zeros);
        double v = labels.size;
        double theta = std::log(w) / std::log(v);
        for (int i = 1; i <= c; i++) {
            double v_i = gsl_vector_int_get(sz, i - 1);
            double w_i = graph_weight(data, labels, nn, ker, i);
            rho += (v_i / v) * (w_i / std::pow(v_i, theta));
        }
    }
    gsl_vector_int_free(sz);
    return rho;
}
