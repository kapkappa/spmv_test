#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <sys/time.h>

#include "helpers.hpp"

#include "cuda_defs.h"

#define FP double
#define CU_FP CUDA_R_64F

#ifdef FP32
#define FP float
#define CU_FP CUDA_R_32F
#endif

//#ifdef FP16
//#define FP float16_t
//#define CU_FP CUDA_R_16F
//#endif

#ifndef CUSPARSE_VERSION
#if defined(CUSPARSE_VER_MAJOR) && defined(CUSPARSE_VER_MINOR) && defined(CUSPARSE_VER_PATCH)
#define CUSPARSE_VERSION (CUSPARSE_VER_MAJOR * 1000 + CUSPARSE_VER_MINOR *  100 + CUSPARSE_VER_PATCH)
#else
#define CUSPARSE_VERSION CUDA_VERSION
#endif
#endif

static inline double timer() {
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char**argv) {

    if (argc != 3) {
        printf("Enter matrix_name and niters\n");
        exit(1);
    }

    FILE *file = fopen(argv[1], "r");

    int nv = 1;

    uint32_t nrows, ncols, nonzeros;
    if (fread(&nrows, sizeof(uint32_t), 1, file) != 1) printf("wrong nrows\n");
    if (fread(&nonzeros, sizeof(uint32_t), 1, file) != 1) printf("wrong nonzeros\n");

    printf("nrows: %d, nonzeros: %d\n", nrows, nonzeros);

    uint32_t *rows=(uint32_t*)calloc(nrows+1, sizeof(uint32_t));
    uint32_t *cols=(uint32_t*)calloc(nonzeros, sizeof(uint32_t));
    double *tmps=(double*)calloc(nonzeros, sizeof(double));
    FP* vals=(FP*)calloc(nonzeros, sizeof(FP));

    if (fread(rows, sizeof(uint32_t), nrows + 1, file) != nrows+1) printf("wrong rows\n");
    if (fread(cols, sizeof(uint32_t), nonzeros, file) != nonzeros) printf("wrong cols\n");
    if (fread(tmps, sizeof(double), nonzeros, file) != nonzeros) printf("wrong vals\n");

    fclose(file);

    for (uint32_t t = 0; t < nonzeros; t++)
        vals[t] = (FP)tmps[t];

    FP *x=(FP*)calloc(nrows, sizeof(FP));
    FP *y=(FP*)calloc(nrows, sizeof(FP));

    set_const<FP>(y, nrows, 0.0);
    set_rand<FP>(x, nrows, nv, ROW);

    FP alpha = 1.0;
    FP beta = 0.0;

    ncols = nrows;

    //--------------------------------------------------------------------------
    printf("Device memory management\n");

    uint32_t   *dA_csrOffsets, *dA_columns;
    FP *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (nrows + 1) * sizeof(uint32_t)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nonzeros * sizeof(uint32_t))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nonzeros * sizeof(FP))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         ncols * sizeof(FP)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         nrows * sizeof(FP)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, rows,
                           (nrows + 1) * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, cols, nonzeros * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, vals, nonzeros * sizeof(FP),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, x, ncols * sizeof(FP),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, y, nrows * sizeof(FP),
                           cudaMemcpyHostToDevice) )

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, nrows, ncols, nonzeros,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CU_FP) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, ncols, dX, CU_FP) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, nrows, dY, CU_FP) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CU_FP,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

#if CUSPARSE_VERSION >= 12400
    CHECK_CUSPARSE( cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CU_FP,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
#endif

    int nwarmups = 5;
    int niters = atoi(argv[2]);

    double dt = 0.0;

    int it, i;
    for (it = 0; it < niters + nwarmups; it++) {

        int loop_iters = 10;

        set_const<FP>(y, nrows, 0.0);
        set_rand<FP>(x, nrows, nv, ROW);
        CHECK_CUDA( cudaMemcpy(dX, x, ncols * sizeof(FP), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dY, y, nrows * sizeof(FP), cudaMemcpyHostToDevice) )

        double t1 = timer();
        for (i = 0; i < loop_iters; i++) {
            CHECK_CUDA( cudaDeviceSynchronize() )
            CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, vecX, &beta, vecY, CU_FP,
                                         CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
        }
        double t2 = timer();

        if (it >= nwarmups) {
            dt += (t2-t1) / loop_iters;
        }
    }

    dt = dt / niters;

    printf("Spmv time: %.12f niters: %d\n", dt, niters);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(y, dY, nrows * sizeof(FP),
                           cudaMemcpyDeviceToHost) )

    print_norm<FP>(y, nrows, 1, ROW);

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )

    free(rows);
    free(cols);
    free(vals);
    free(tmps);

    return EXIT_SUCCESS;
}
