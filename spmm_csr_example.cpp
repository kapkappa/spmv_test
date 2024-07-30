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

    if (argc != 4) {
        printf("Enter matrix name, niters and NV\n");
        exit(1);
    }

    FILE *file = fopen(argv[1], "r");

    int nv = atoi(argv[3]);
    printf("nv=%d\n", nv);

    uint32_t nrows, ncols, nonzeros;
    if (fread(&nrows, sizeof(uint32_t), 1, file) != 1) printf("wrong nrows\n");
    if (fread(&nonzeros, sizeof(uint32_t), 1, file) != 1) printf("wrong nonzeros\n");

    printf("nrows: %d, nonzeros: %d\n", nrows, nonzeros);
    ncols = nrows;

    uint32_t *rows=(uint32_t*)calloc(nrows+1, sizeof(uint32_t));
    uint32_t *cols=(uint32_t*)calloc(nonzeros, sizeof(uint32_t));
    double *tmps=(double*)calloc(nonzeros, sizeof(double));
    FP *vals=(FP*)calloc(nonzeros, sizeof(FP));

    if (fread(rows, sizeof(uint32_t), nrows + 1, file) != nrows+1) printf("wrong rows\n");
    if (fread(cols, sizeof(uint32_t), nonzeros, file) != nonzeros) printf("wrong cols\n");
    if (fread(tmps, sizeof(double), nonzeros, file) != nonzeros) printf("wrong vals\n");

    fclose(file);

    for (uint32_t t = 0; t < nonzeros; t++)
        vals[t] = (FP)tmps[t];

    int vec_size = ncols * nv;

    FP *x=(FP*)calloc(vec_size, sizeof(FP));
    FP *y=(FP*)calloc(vec_size, sizeof(FP));

    set_const<FP>(y, vec_size, 0.0);
    set_rand<FP>(x, ncols, nv, ROW);

    FP alpha = 1.0;
    FP beta = 0.0;


    //--------------------------------------------------------------------------
    printf("Device memory management\n");

    uint32_t   *dA_csrOffsets, *dA_columns;
    FP *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (nrows + 1) * sizeof(uint32_t)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nonzeros * sizeof(uint32_t))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nonzeros * sizeof(FP))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         vec_size * sizeof(FP)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         vec_size * sizeof(FP)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, rows,
                           (nrows + 1) * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, cols, nonzeros * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, vals, nonzeros * sizeof(FP),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, x, vec_size * sizeof(FP),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, y, vec_size * sizeof(FP),
                           cudaMemcpyHostToDevice) )

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matX, matY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, nrows, ncols, nonzeros,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CU_FP) )
    // Create dense matrix X
    CHECK_CUSPARSE( cusparseCreateDnMat(&matX, ncols, nv, nv, dX, CU_FP, CUSPARSE_ORDER_ROW) )
    // Create dense matrix y
    CHECK_CUSPARSE( cusparseCreateDnMat(&matY, nrows, nv, nv, dY, CU_FP, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matX, &beta, matY, CU_FP,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )


#if CUSPARSE_VERSION >= 12400
    CHECK_CUSPARSE( cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matX, &beta, matY, CU_FP,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
#endif

    int nwarmups = 5;
    int niters = atoi(argv[2]);

    double dt = 0.0;

    int it, i;
    for (it = 0; it < niters + nwarmups; it++) {

        int loop_iters = 10;

        set_const<FP>(y, vec_size, 0.0);
        set_rand<FP>(x, ncols, nv, ROW);
        CHECK_CUDA( cudaMemcpy(dX, x, vec_size * sizeof(FP), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dY, y, vec_size * sizeof(FP), cudaMemcpyHostToDevice) )

        double t1 = timer();
        for (i = 0; i < loop_iters; i++) {
            CHECK_CUDA( cudaDeviceSynchronize() )
            CHECK_CUSPARSE( cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, matX, &beta, matY, CU_FP,
                                         CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
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
    CHECK_CUSPARSE( cusparseDestroyDnMat(matX) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(y, dY, vec_size * sizeof(FP),
                           cudaMemcpyDeviceToHost) )


    print_norm<FP>(y, nrows, nv, ROW);


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
