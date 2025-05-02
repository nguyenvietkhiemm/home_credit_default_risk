#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

__global__ void interest_rate_kernel(double* amt_annuity, double* amt_credit, int* cnts, double* rates, int size, double tol, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        double temp = amt_annuity[idx] / amt_credit[idx];
        double r = 0.9;
        int cnt = cnts[idx];

        for (int j = 0; j < max_iter; ++j) {
            double pow_r = pow(1.0 + r, cnt);
            if (pow_r == 0.0) break;

            double new_r = temp * (pow_r - 1.0) / pow_r;

            if (abs(new_r - r) < tol) {
                r = new_r;
                break;
            }

            r = new_r;
        }

        rates[idx] = r;
    }
}

extern "C" void interest_rate(double* amt_annuity, double* amt_credit, int* cnts, double* rates, int size, double tol=1e-7, int max_iter=500) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    double* d_amt_annuity, * d_amt_credit, * d_rates;
    int* d_cnts;

    cudaMalloc(&d_amt_annuity, size * sizeof(double));
    cudaMalloc(&d_amt_credit, size * sizeof(double));
    cudaMalloc(&d_cnts, size * sizeof(int));
    cudaMalloc(&d_rates, size * sizeof(double));

    cudaMemcpy(d_amt_annuity, amt_annuity, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amt_credit, amt_credit, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnts, cnts, size * sizeof(int), cudaMemcpyHostToDevice);

    interest_rate_kernel<<<numBlocks, blockSize>>>(d_amt_annuity, d_amt_credit, d_cnts, d_rates, size, tol, max_iter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(rates, d_rates, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_amt_annuity);
    cudaFree(d_amt_credit);
    cudaFree(d_cnts);
    cudaFree(d_rates);
}
