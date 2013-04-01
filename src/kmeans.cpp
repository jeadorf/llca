#include "llca/kmeans.hpp"

#include <KMlocal.h>

gsl_vector_int* llca::kmeans(const gsl_matrix &data, int c) {
    gsl_vector_int *labels = gsl_vector_int_alloc(data.size1);

    KMdata dataPts(data.size2, data.size1);
    KMdataArray kmPts = dataPts.getPts();
    KMterm term(100, 0, 0, 0,
             0.10, 0.10, 3,
             0.50, 10, 0.95);
    for(size_t i = 0; i < data.size1; i++) {
      for(size_t j = 0; j < data.size2; j++) {
          kmPts[i][j] = gsl_matrix_get(&data, i, j);
      }
    }

    dataPts.buildKcTree();
    KMfilterCenters *ctrs = new KMfilterCenters(c, dataPts);
    KMlocalLloyds kmAlg(*ctrs, term);
    *ctrs = kmAlg.execute();

    KMctrIdxArray closeCtr = new KMctrIdx[dataPts.getNPts()];
    double* sqDist = new double[dataPts.getNPts()];
    ctrs->getAssignments(closeCtr, sqDist);
    for(size_t i = 0; i < data.size1; i++) {
        gsl_vector_int_set(labels, i, closeCtr[i] + 1);
    }

    delete [] sqDist;
    delete [] closeCtr;
    delete ctrs;

    return labels;
}
