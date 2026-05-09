#include "block_gemm_omp.h"
#include <algorithm>

/* Optimizations
1. Simple block version
*/

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n);

    int block_size = 32;

    for (int block_i = 0; block_i < n / block_size; block_i++) {
        for (int block_j = 0; block_j < n / block_size; block_j++) {
            for (int block_k = 0; block_k < n / block_size; block_k++) {
                // Calculating borders
                int i_left = block_i * block_size;
                int i_right = i_left + block_size;
                int j_left = block_j * block_size;
                int j_right = j_left + block_size;
                int k_left = block_k * block_size;
                int k_right = k_left + block_size;

                for (int i = i_left; i < i_right; i++) {
                    for (int k = k_left; k < k_right; k++) {
                        for (int j = j_left; j < j_right; j++) {
                            c[i * n + j] += a[i * n + k] * b[k * n + j];
                        }
                    }
                }
            }
        }
    }



    return c;
}