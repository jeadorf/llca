#ifndef LLCA_KNN_HPP
#define	LLCA_KNN_HPP

#include <gsl/gsl_matrix.h>

namespace llca {

class NearestNeighbors {

protected:
    
    int k_;
    
public:
    
    NearestNeighbors(int k) : k_(k) {}
    
    virtual double operator()(int i, int j) const = 0;
    
    int get_k() const { return k_; }
    
};
    
class BruteForceNearestNeighbors : public NearestNeighbors {
    
    int *nn_;
    
public:
    
    BruteForceNearestNeighbors(const gsl_matrix *data, int k);
    
    ~BruteForceNearestNeighbors();
    
    double operator()(int i, int j) const;
    
};

}

#endif	/* LLCA_KNN_HPP */
