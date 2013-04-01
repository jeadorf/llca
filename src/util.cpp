#include "llca/util.hpp"

#include <boost/format.hpp>
#include <ctime>
#include <fstream>
#include <sstream>

std::string llca::create_tmp_dir() {
    time_t t = time(NULL);
    struct tm *ltm = localtime(&t);
    char templ[64];
    strftime(templ, 64, "/tmp/clustc_llca_%Y-%m-%d_%H-%M-%S_XXXXXX", ltm);
    if (mkdtemp(templ)) {
        return std::string(templ);
    } else {
        // Fail. This case is too dangerous to handle.
        throw std::runtime_error("Could not create temporary directory.");
    }
}

std::string llca::save_indicator(const std::string &result_dir,
                                    const gsl_vector_int &indicator) {
    std::ofstream o;
    std::string f = str(boost::format("%s/indicator") % result_dir);
    o.open(f.c_str());
    for (size_t i = 0; i < indicator.size; i++) {
        o << gsl_vector_int_get(&indicator, i) << std::endl;
    }
    o.close();
    return f;
}

void gsl_matrix_dlmwrite(const std::string &filename,
                            const gsl_matrix &matrix) {
    std::ofstream o;
    o.open(filename.c_str());
    for (size_t i = 0; i < matrix.size1; i++) {
        for (size_t j = 0; j < matrix.size2 - 1; j++) {
            o << gsl_matrix_get(&matrix, i, j) << " ";
        }
        o << gsl_matrix_get(&matrix, i, matrix.size2 - 1) << std::endl;
    }
    o.close();
}

void llca::save_workspace(const std::string &result_dir,
                            const llca::Workspace &ws) {
    gsl_matrix_dlmwrite(str(boost::format("%s/A") % result_dir), *ws.A);
    gsl_matrix_dlmwrite(str(boost::format("%s/T") % result_dir), *ws.T);
    gsl_matrix_dlmwrite(str(boost::format("%s/scores") % result_dir), *ws.scores);
}

gsl_matrix* llca::read_matrix(const std::string &file) {
    size_t c = 256;
    size_t s = 0;
    size_t n = 0;
    double *d = new double[c];
    std::ifstream in;
    in.open(file.c_str());
    while (in.good()) {
        std::string line;
        std::getline(in, line);
        std::stringstream lin;
        lin << line;
        lin >> std::ws;
        if (lin.str() != "") {
            while (lin.good()) {
                if (s == c) {
                    c *= 2;
                    double *tmp = new double[c];
                    std::copy(d, d + s, tmp);
                    delete [] d;
                    d = tmp;
                }
                lin >> d[s++];
                lin >> std::ws;
            }
            n++;
        }
    }
    in.close();
    if (n == 0) {
        throw std::runtime_error("could not read matrix");
    }
    gsl_matrix *m = gsl_matrix_alloc(n, s / n);
    std::copy(d, d + s, m->data);
    delete [] d;
    return m;
}

gsl_vector_int* llca::read_vector_int(const std::string &file) {
    size_t c = 256;
    size_t s = 0;
    double *d = new double[c];
    std::ifstream in;
    in.open(file.c_str());
    while (in.good()) {
        if (s == c) {
            c *= 2;
            double *tmp = new double[c];
            std::copy(d, d + s, tmp);
            delete [] d;
            d = tmp;
        }
        in >> std::ws;
        in >> d[s++];
        in >> std::ws;
    }
    in.close();
    gsl_vector_int *v = gsl_vector_int_alloc(s);
    std::copy(d, d + s, v->data);
    delete [] d;
    return v;
}
