#ifndef LLCA_SVM_HPP
#define	LLCA_SVM_HPP

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <svm.h>

namespace llca {

class SVM {

    svm_parameter param;

    svm_problem problem;

    svm_model *model;

  public:

    SVM(double gamma, double cost);

    ~SVM();

    void train(const gsl_matrix& data, const gsl_vector_int &labels);

    gsl_vector_int* predict(const gsl_matrix& data) const;

};

}

#endif /* LLCA_SVM_HPP */
