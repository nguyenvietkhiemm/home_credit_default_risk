#include <cmath>
#include <cstdlib> 
extern "C" void interest_rate(double* amt_annuity, double* amt_credit, int* cnts, double* rates, int size, double tol=1e-7, int max_iter=500) {
    for (int i = 0; i < size; ++i) {
        double temp = amt_annuity[i] / amt_credit[i];
        double r = 0.9;
        int cnt = cnts[i];

        for (int j = 0; j < max_iter; ++j) {
            double pow_r = std::pow(1.0 + r, cnt);
            if (pow_r == 0.0) break;

            double new_r = temp * (pow_r - 1.0) / pow_r;

            if (std::abs(new_r - r) < tol) {
                r = new_r;
                break;
            }

            r = new_r;
        }

        rates[i] = r;
    }
}
