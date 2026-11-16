#pragma once
// Apple Silicon AMX Coprocessor (`Accelerate` LAPACK) Float64 Symmetric Eigensolver.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace mps_kit {

// Computes ascending eigenvalues `w[0..n-1]` and orthonormal eigenvectors in column-major `A [n x n]`
// using divide-and-conquer `dsyevd_` with automatic QR iteration fallback (`dsyev_`).
inline void amx_sym_eigh_f64(int n, double* A, double* w) {
    if (n <= 0) return;
    if (n == 1) {
        w[0] = A[0];
        A[0] = 1.0;
        return;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) {
            double avg = 0.5 * (A[j * n + i] + A[i * n + j]);
            A[j * n + i] = avg;
            A[i * n + j] = avg;
        }
    }
    std::vector<double> backup(A, A + static_cast<size_t>(n) * static_cast<size_t>(n));
    char jobz = 'V';
    char uplo = 'U';
    __CLPK_integer nn = n;
    __CLPK_integer lda = n;
    __CLPK_integer info = 0;
    double wkopt = 0.0;
    __CLPK_integer iwkopt = 0;
    __CLPK_integer lwork = -1;
    __CLPK_integer liwork = -1;

    dsyevd_(&jobz, &uplo, &nn, A, &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info);
    lwork = std::max<__CLPK_integer>(static_cast<__CLPK_integer>(wkopt) + 32, 1 + 6 * nn + 2 * nn * nn);
    liwork = std::max<__CLPK_integer>(iwkopt + 16, 3 + 5 * nn);
    std::vector<double> work(static_cast<size_t>(lwork));
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(liwork));
    dsyevd_(&jobz, &uplo, &nn, A, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info == 0) {
        return;
    }

    std::memcpy(A, backup.data(), backup.size() * sizeof(double));
    __CLPK_integer lwork2 = std::max<__CLPK_integer>(3 * nn, 64);
    std::vector<double> work2(static_cast<size_t>(lwork2));
    info = 0;
    dsyev_(&jobz, &uplo, &nn, A, &lda, w, work2.data(), &lwork2, &info);
    if (info != 0) {
        std::fill(w, w + n, 0.0);
        std::fill(A, A + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            A[i * n + i] = 1.0;
        }
    }
}

} // namespace mps_kit
