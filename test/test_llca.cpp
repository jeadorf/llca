#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_llca

#include "llca/llca.hpp"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "llca/configure.hpp"
#include "llca/kernel.hpp"
#include "llca/knn.hpp"
#include "llca/util.hpp"

BOOST_AUTO_TEST_CASE(TinyExample) {
    gsl_matrix *data = gsl_matrix_calloc(10, 2);
    int c = 2;
    double sigma = 1;
    int knn = 5;
    double lambda = 1;
    llca::Workspace workspace(data->size1, c);
    gsl_vector_int *indicator = llca::cluster(*data, c, sigma, knn, lambda, workspace);
    gsl_vector_int_free(indicator);
    gsl_matrix_free(data);
}

BOOST_AUTO_TEST_CASE(RbfKernel) {
    gsl_matrix *X = llca::read_matrix(llca::get_resource("data/test_kernel_X"));
    gsl_matrix *K = llca::read_matrix(llca::get_resource("data/test_kernel_K"));
    
    // Simple and stupid sample for testing whether the data was read in correctly
    BOOST_CHECK_CLOSE(gsl_matrix_get(K, 0, 0), 1.0, 1e-6);
    
    llca::RbfKernel ker(0.5);
    
    for (size_t i = 0; i < X->size1; i++) {
        for (size_t j = 0; j < X->size1; j++) {
            BOOST_CHECK_CLOSE(gsl_matrix_get(K, i, j), ker(*X, i, j), 1e-5);
        }
    }

    gsl_matrix_free(X);
    gsl_matrix_free(K);
}

BOOST_AUTO_TEST_CASE(BruteForceNearestNeighbors) {
    gsl_matrix *X = llca::read_matrix(llca::get_resource("data/test_knn_X"));
    gsl_matrix *N = llca::read_matrix(llca::get_resource("/data/test_knn_N"));
  
    BOOST_CHECK_EQUAL(10, N->size2);
    llca::BruteForceNearestNeighbors nn(X, N->size2);
    
    for (size_t i = 0; i < N->size1; i++) {
        for (size_t j = 0; j < N->size2; j++) {
            BOOST_CHECK_EQUAL(gsl_matrix_get(N, i, j), nn(i, j));
        }
    }

    gsl_matrix_free(X);
    gsl_matrix_free(N);
}

BOOST_AUTO_TEST_CASE(TwoMoons) {
    gsl_matrix *Xh = llca::read_matrix(llca::get_resource("data/twomoons500train.txt"));
    gsl_matrix_view X = gsl_matrix_submatrix(Xh, 0, 0, Xh->size1, 2);
    gsl_matrix *A = llca::read_matrix(llca::get_resource("data/test_A"));
    gsl_matrix *T = llca::read_matrix(llca::get_resource("data/test_T"));
    gsl_matrix *scores = llca::read_matrix(llca::get_resource("data/test_scores"));
    gsl_vector_int *indicator = llca::read_vector_int(llca::get_resource("data/test_indicator"));
    
    BOOST_CHECK_EQUAL(2, X.matrix.size2);
    
    int c = 2;
    llca::Workspace workspace(X.matrix.size1, c);
    gsl_vector_int *ind_act = llca::cluster(X.matrix, c, 0.1, 10, 0.1, workspace);
    
    for (size_t i = 0; i < X.matrix.size1; i++) {
        for (size_t j = 0; j < X.matrix.size2; j++) {
            BOOST_CHECK_CLOSE(gsl_matrix_get(A, i, j), gsl_matrix_get(workspace.A, i, j), 1e-2);
            BOOST_CHECK_CLOSE(gsl_matrix_get(T, i, j), gsl_matrix_get(workspace.T, i, j), 1e-2);
        }
        for (int j = 0; j < c; j++) {
            BOOST_CHECK_CLOSE(gsl_matrix_get(scores, i, j), gsl_matrix_get(workspace.scores, i, j), 1e-2);
        }
    }
    
    int s = 0;
    for (size_t i = 0; i < X.matrix.size1; i++) {
        s += gsl_vector_int_get(indicator, i);
        s -= gsl_vector_int_get(ind_act, i);
    }
    BOOST_CHECK(s <= int(X.matrix.size1) && (s >= int(X.matrix.size1) - 3 || s <= 3));
   
    gsl_matrix_free(Xh);
    gsl_matrix_free(A);
    gsl_matrix_free(T);
    gsl_matrix_free(scores);
    gsl_vector_int_free(indicator);
    gsl_vector_int_free(ind_act);
}

BOOST_AUTO_TEST_CASE(TwoMoonsWithOos) {
    gsl_matrix *Xh = llca::read_matrix(llca::get_resource("data/twomoons500train.txt"));
    gsl_matrix_view X = gsl_matrix_submatrix(Xh, 0, 0, Xh->size1, 2);
    gsl_vector_int *ind_e = llca::read_vector_int(llca::get_resource("data/test_indicator"));
    srand(125752824);
    gsl_vector_int *ind_a = llca::cluster_with_oosext(X.matrix, 2, 1, 10, 0.1, 0.75, 10, 1);
    int s = 0;
    for (size_t i = 0; i < X.matrix.size1; i++) {
        s += gsl_vector_int_get(ind_e, i);
        s -= gsl_vector_int_get(ind_a, i);
    }
    BOOST_CHECK(s >= 0 && s <= int(X.matrix.size1 + 15) && (s >= int(X.matrix.size1) - 15 || s <= 15));
    
    gsl_vector_int_free(ind_a);
    gsl_vector_int_free(ind_e);
    gsl_matrix_free(Xh);
}

BOOST_AUTO_TEST_CASE(create_tmp_dir) {
    std::string tmpdir = llca::create_tmp_dir();
    BOOST_CHECK(boost::filesystem::is_directory(tmpdir));
    BOOST_CHECK(boost::filesystem::exists(tmpdir));
    BOOST_CHECK(boost::filesystem::is_empty(tmpdir));
    if (boost::filesystem::is_empty(tmpdir)) {
        boost::filesystem::remove_all(tmpdir);
        BOOST_CHECK( ! boost::filesystem::is_directory(tmpdir));
        BOOST_CHECK( ! boost::filesystem::exists(tmpdir));
    }
}

BOOST_AUTO_TEST_CASE(save_indicator) {
    std::string tmpdir = llca::create_tmp_dir();
    BOOST_CHECK(boost::filesystem::is_empty(tmpdir));
    
    const int n = 100;
    gsl_vector_int *indicator = gsl_vector_int_alloc(n);
    for (int i = 0; i < n; i++) {
        gsl_vector_int_set(indicator, i, 1 + (i * i) % 3);
    }
    
    std::string f = llca::save_indicator(tmpdir, *indicator);
    BOOST_CHECK(! boost::filesystem::is_empty(tmpdir));
    gsl_vector_int *indicator2 = llca::read_vector_int(f);
    for (int i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(gsl_vector_int_get(indicator, i), gsl_vector_int_get(indicator2, i));
    }
    gsl_vector_int_free(indicator);
    gsl_vector_int_free(indicator2);
    boost::filesystem::remove_all(tmpdir);
}

BOOST_AUTO_TEST_CASE(save_workspace) {
    gsl_matrix *data = llca::read_matrix(llca::get_resource("data/twomoons500train.txt"));
    gsl_matrix_view data_view = gsl_matrix_submatrix(data, 0, 0, data->size1 / 5, data->size2);
    const int c = 2;
    const int n = data_view.matrix.size1;
    llca::Workspace workspace(n, c);
    gsl_vector_int *ind = llca::cluster(data_view.matrix, c, 0.1, 10, 0.1, workspace);
    gsl_vector_int_free(ind);
    std::string tmpdir = llca::create_tmp_dir();
    llca::save_workspace(tmpdir, workspace);
    
    gsl_matrix *A = llca::read_matrix(str(boost::format("%s/A") % tmpdir));
    gsl_matrix *T = llca::read_matrix(str(boost::format("%s/T") % tmpdir));
    gsl_matrix *scores = llca::read_matrix(str(boost::format("%s/scores") % tmpdir));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            BOOST_CHECK_CLOSE(gsl_matrix_get(A, i, j), gsl_matrix_get(workspace.A, i, j), 1e-2);
            BOOST_CHECK_CLOSE(gsl_matrix_get(T, i, j), gsl_matrix_get(workspace.T, i, j), 1e-2);
        }
        for (int j = 0; j < c; j++) {
            BOOST_CHECK_CLOSE(gsl_matrix_get(scores, i, j), gsl_matrix_get(workspace.scores, i, j), 1e-2);
        }
    }
    
    gsl_matrix_free(data);
    gsl_matrix_free(A);
    gsl_matrix_free(T);
    gsl_matrix_free(scores);
    boost::filesystem::remove_all(tmpdir);
}
