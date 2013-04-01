#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_util

#include "llca/util.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "llca/configure.hpp"

BOOST_AUTO_TEST_CASE(read_matrix) {
    gsl_matrix *m = llca::read_matrix(llca::get_resource("data/test_read_matrix"));
    BOOST_CHECK_EQUAL(4, m->size1);
    BOOST_CHECK_EQUAL(3, m->size2);
    BOOST_CHECK_EQUAL(12, gsl_matrix_get(m, 0, 0));
    BOOST_CHECK_EQUAL(43, gsl_matrix_get(m, 0, 1));
    BOOST_CHECK_EQUAL(22, gsl_matrix_get(m, 0, 2));
    BOOST_CHECK_EQUAL(83, gsl_matrix_get(m, 3, 0));
    BOOST_CHECK_EQUAL(29, gsl_matrix_get(m, 3, 1));
    BOOST_CHECK_EQUAL(45, gsl_matrix_get(m, 3, 2));
    gsl_matrix_free(m);
}

BOOST_AUTO_TEST_CASE(read_vector_int) {
    gsl_vector_int *v = llca::read_vector_int(llca::get_resource("data/test_read_vector_int"));
    BOOST_CHECK_EQUAL(4, v->size);
    BOOST_CHECK_EQUAL(12, gsl_vector_int_get(v, 0));
    BOOST_CHECK_EQUAL(43, gsl_vector_int_get(v, 1));
    BOOST_CHECK_EQUAL(22, gsl_vector_int_get(v, 2));
    BOOST_CHECK_EQUAL(45, gsl_vector_int_get(v, 3));
    gsl_vector_int_free(v);
}
