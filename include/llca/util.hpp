#ifndef CLUSTC_UTIL_HPP
#define	CLUSTC_UTIL_HPP

#include "llca.hpp"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <string>

namespace llca {

std::string create_tmp_dir();
    
std::string save_indicator(const std::string &result_dir,
                            const gsl_vector_int &indicator);

void save_workspace(const std::string &result_dir,
                        const llca::Workspace &ws);

gsl_matrix* read_matrix(const std::string &file);

gsl_vector_int* read_vector_int(const std::string &file);

}

#endif	/* CLUSTC_UTIL_HPP */
