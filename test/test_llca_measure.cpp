#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_kmeansSpectral

#include "llca/measure.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cmath>

// ----------------------------------------------------------------------------.
// Cluster size
// ----------------------------------------------------------------------------.

BOOST_AUTO_TEST_CASE(cluster_size) {
    int labelsv[] = { 1, 1, 1, 2, 2, 3 };
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    gsl_vector_int *sz = llca::measure::cluster_sizes(labels.vector, 3);
    BOOST_CHECK_EQUAL(3, gsl_vector_int_get(sz, 0));
    BOOST_CHECK_EQUAL(2, gsl_vector_int_get(sz, 1));
    BOOST_CHECK_EQUAL(1, gsl_vector_int_get(sz, 2));
    gsl_vector_int_free(sz);
}

// ----------------------------------------------------------------------------.
// Cluster balance
// ----------------------------------------------------------------------------.

BOOST_AUTO_TEST_CASE(cluster_balance_perfect) {
    int labelsv[] = { 1, 1, 2, 2, 3, 3 };
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    BOOST_CHECK_CLOSE(1.0, llca::measure::cluster_balance(labels.vector, 3), 1e-9);
}

BOOST_AUTO_TEST_CASE(cluster_balance_worst_case) {
    int labelsv[] = { 2, 2, 2, 2, 2, 2 };
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    BOOST_CHECK_CLOSE(0.0, llca::measure::cluster_balance(labels.vector, 3), 1e-9);
}

BOOST_AUTO_TEST_CASE(cluster_balance_quarter) {
    int labelsv[] = { 1, 1, 1, 1, 2, 3 };
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    BOOST_CHECK_CLOSE(0.25, llca::measure::cluster_balance(labels.vector, 3), 1e-9);
}

BOOST_AUTO_TEST_CASE(cluster_balance_half) {
    int labelsv[] = { 1, 1, 1, 1, 2, 2 };
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    BOOST_CHECK_CLOSE(0.5, llca::measure::cluster_balance(labels.vector, 2), 1e-9);
}

// ----------------------------------------------------------------------------.
// Expected density
// ----------------------------------------------------------------------------.

BOOST_AUTO_TEST_CASE(expected_density_worst_case) {
    double datam[] = { 1, 3, 5, 7, 9, 4 }; 
    int labelsv[] = { 2, 2, 2, 2, 2, 2 };
    gsl_matrix_view data = gsl_matrix_view_array(datam, 6, 1);
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    llca::RbfKernel ker(1.0);
    BOOST_CHECK_CLOSE(0.0, llca::measure::expected_density(data.matrix, labels.vector, 3, ker, 5), 1e-9);
}

BOOST_AUTO_TEST_CASE(expected_density_best_clustering_1d) {
    double datam[] = { -2.75, -1.75, -0.75, 0.75, 1.75, 2.75 }; 
    int labels_goodv[] = { 1, 1, 1, 2, 2, 2 };
    int labels_badv[] = { 1, 1, 2, 2, 2, 2 };
    gsl_matrix_view data = gsl_matrix_view_array(datam, 6, 1);
    gsl_vector_int_view labels_good = gsl_vector_int_view_array(labels_goodv, 6);
    gsl_vector_int_view labels_bad = gsl_vector_int_view_array(labels_badv, 6);
    llca::RbfKernel ker(1.0);
    BOOST_CHECK_GT(llca::measure::expected_density(data.matrix, labels_good.vector, 2, ker, 2),
                    llca::measure::expected_density(data.matrix, labels_bad.vector, 2, ker, 2));
}

BOOST_AUTO_TEST_CASE(expected_density_best_clustering_1d_v2) {
    double datam[] = { -3, -2, -1, 1, 2, 3 }; 
    int labels_goodv[] = { 1, 1, 1, 2, 2, 2 };
    int labels_badv[] = { 1, 1, 2, 2, 2, 2 };
    gsl_matrix_view data = gsl_matrix_view_array(datam, 6, 1);
    gsl_vector_int_view labels_good = gsl_vector_int_view_array(labels_goodv, 6);
    gsl_vector_int_view labels_bad = gsl_vector_int_view_array(labels_badv, 6);
    llca::RbfKernel ker(1.0);
    BOOST_CHECK_GT(llca::measure::expected_density(data.matrix, labels_good.vector, 2, ker, 2),
                    llca::measure::expected_density(data.matrix, labels_bad.vector, 2, ker, 2));
}

BOOST_AUTO_TEST_CASE(expected_density_exact) {
    double datam[] = {  -2.75, -1.75, -0.75, 0.75, 1.75, 2.75 }; 
    int labelsv[] = { 1, 1, 1, 2, 2, 2 };
    gsl_matrix_view data = gsl_matrix_view_array(datam, 6, 1);
    gsl_vector_int_view labels = gsl_vector_int_view_array(labelsv, 6);
    double e1 = std::exp(-1);
    double e4 = std::exp(-4);
    double e2_25 = std::exp(-2.25);
    // see Definition 6 in  [1] B. Stein and S. Meyer, “On Cluster Validity and the Information Need of
    // Users,” in 3rd International Conference on Artificial Intelligence and
    // Applications, 2003, pp. 216–221.
    double expected = 2 * (3.0/6.0) * ((2 * e1 + e4)/std::pow(3.0, std::log(4 * e1 + 2 * e4 + e2_25) / std::log(6)));
    BOOST_CHECK_CLOSE(0.5624, expected, 1e-2);
    llca::RbfKernel ker(1.0);
    BOOST_CHECK_CLOSE(expected, llca::measure::expected_density(data.matrix, labels.vector, 2, ker, 2), 1e-2);
}
