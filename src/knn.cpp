#include "llca/knn.hpp"

#include <boost/format.hpp>
#include <set>
#include <stdexcept>
#include <vector>

typedef std::pair<double, int> neighbor;
    
llca::BruteForceNearestNeighbors::BruteForceNearestNeighbors(const gsl_matrix *data, int k) : NearestNeighbors(k) {
    nn_ = new int[data->size1 * k_];
    int n = data->size1;

    if (n <= k_ + 1) {
        throw std::runtime_error(str(boost::format("size of dataset (%d) is too small for the number of nearest neighbors (%d)") % n % k));
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::set<neighbor> nbh;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double d = 0;
                double c = 0;
                for (size_t l = 0; l < data->size2; l++) {
                    c = gsl_matrix_get(data, i, l) - gsl_matrix_get(data, j, l);
                    d += c * c;
                }
                nbh.insert(std::make_pair(d, j));
            }
        }
        
        
        assert(nbh.size() >= size_t(k_));

        std::set<neighbor>::iterator nb = nbh.begin();
        for (int j = 0; j < k_; j++, nb++) {
            nn_[i * k_ + j] = nb->second;
        }
    }
}

llca::BruteForceNearestNeighbors::~BruteForceNearestNeighbors() {
    delete [] nn_;
}
    
double llca::BruteForceNearestNeighbors::operator()(int i, int j) const {
    return nn_[i * k_ + j];
}
