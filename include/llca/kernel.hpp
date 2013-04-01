#ifndef LLCA_KERNEL_HPP
#define	LLCA_KERNEL_HPP

#include <gsl/gsl_matrix.h>

namespace llca {
    
class Kernel {
    
public:
    
    virtual double operator()(const gsl_matrix &data, int i, int j) const = 0;

};

class RbfKernel : public Kernel {

    double sigma_;

public:

    RbfKernel(double sigma) : sigma_(sigma) {}

    double operator()(const gsl_matrix &data, int i, int j) const;

};

}

#endif	/* LLCA_KERNEL_HPP */

