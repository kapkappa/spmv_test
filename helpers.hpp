#include <sys/types.h>
#include <climits>
#include <stdio.h>
#include <iostream>

namespace {

static inline uint32_t xorshift32(uint32_t i) {
    uint32_t state = i + 1;
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

static inline int fastrand(int g_seed) {
    g_seed = 1664525 * g_seed + 1013904223;
    return g_seed;
}

static inline int int_pseudo_rand(uint32_t i) {
    return fastrand(xorshift32(i));
}

static inline double dbl_pseudo_rand(const uint64_t i) {
    double val = int_pseudo_rand(i);
    return 0.5 * (1.0 + val / INT_MAX);
}

} // namespace


enum type_t {COL, ROW};


template <typename T>
void set_const(T* vec, size_t size, T val) {
    for (size_t i = 0; i < size; i++)
        vec[i] = val;
}


template <typename T>
void set_rand(T* vec, size_t size, size_t nv, type_t order) {
    size_t i, j;

    if (order == COL) {   //column-wise order
        for (j = 0; j < nv; j++) {
            for (i = 0; i < size; i++) {
                vec[j*size + i] = (T)dbl_pseudo_rand(j + i*nv) - 0.5;
            }
        }
    } else if (order == ROW) {              //row-wise order
        for (i = 0; i < nv*size; i+=nv) {
            for (j = 0; j < nv; j++) {
                vec[i+j] = (T)dbl_pseudo_rand(i+j) - 0.5;
            }
        }
    } else {
        std::cout << "Wrong order" << std::endl;
    }
}


template <typename T>
void print_norm(T* vec, size_t size, size_t nv, type_t order) {
    double norm[nv];
    for (size_t i = 0; i < nv; i++)
        norm[i] = 0.0;

    if (order == COL) {   //column-wise order
        for (size_t j = 0; j < nv; j++) {
            for (size_t i = 0; i < size; i++) {
                norm[j] += vec[j*size + i] * vec[j*size + i];
            }
        }
    } else if (order == ROW) {            //row-wise order
        for (size_t i = 0; i < nv*size; i+=nv) {
            for (size_t j = 0; j < nv; j++) {
                norm[j] += vec[j + i] * vec[j + i];
            }
        }
    } else {
        std::cout << "Wrong order" << std::endl;
    }
    printf("vector norm: ");
    for (size_t i = 0; i < nv; i++) {
        norm[i] = sqrt(norm[i]);
        printf("| %.12e |", norm[i]);
    }
    printf("\n");
}
