#include "llca/svm.hpp"

#include <stdexcept>

llca::SVM::SVM(double gamma, double cost) : param(), problem(), model(NULL) {
    param.svm_type = C_SVC;
	param.kernel_type = RBF;
    param.gamma = gamma;
	param.cache_size = 100;
	param.eps = 0.001;
	param.C = cost;
	param.nr_weight = 0;
	param.shrinking = 0;
	param.probability = 0;

    problem.l = 0;
}

llca::SVM::~SVM() {
    delete [] problem.y;
    for (int i = 0; i < problem.l; i++) {
        free(problem.x[i]);
    }
    free(problem.x);
    svm_free_and_destroy_model(&model);
}

void svm_print_null(const char *s) {};

void llca::SVM::train(const gsl_matrix& data, const gsl_vector_int &labels) {
    delete [] problem.y;
    delete model;

    const char *error_msg = svm_check_parameter(&problem, &param);
    if (error_msg) {
        throw std::runtime_error(error_msg);
    }

    //number of training data
    problem.l = labels.size;
    //point to array of classes
    problem.y = new double[labels.size];
    for (size_t i = 0; i < labels.size; i++) {
        problem.y[i] = double(gsl_vector_int_get(&labels, i));
    }
    //generate sparse representation of train data (see libSVM README)
    problem.x = (svm_node**) malloc(sizeof(svm_node*) * data.size1);
    for(size_t i = 0; i < data.size1; i++) {
        problem.x[i] = (svm_node*) malloc(sizeof(svm_node) * (data.size2 + 1));
        for(size_t j = 0; j < data.size2; j++) {
            problem.x[i][j].index = j;
            problem.x[i][j].value = gsl_matrix_get(&data, i, j);
        }
        problem.x[i][data.size2].index = -1; //indicate eol
    }

    svm_set_print_string_function(&svm_print_null); // global!
    model = svm_train(&problem, &param);
}

gsl_vector_int* llca::SVM::predict(const gsl_matrix& data) const {
    gsl_vector_int *labels = gsl_vector_int_alloc(data.size1);
    svm_node *s_node = new svm_node[data.size2 + 1];
    for (size_t i = 0; i < data.size1; i++) {
        for (size_t j = 0; j < data.size2; j++) {
          s_node[j].index = j;
          s_node[j].value = gsl_matrix_get(&data, i, j);
        }
        s_node[data.size2].index = -1;
        gsl_vector_int_set(labels, i, svm_predict(model, s_node));
    }
    delete [] s_node;
    return labels;
}
