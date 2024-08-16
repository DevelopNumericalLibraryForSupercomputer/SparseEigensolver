#ifndef MATVEC_WRAPPER_H
#define MATVEC_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

void c_matvec_wrapper(double* arr, int rows, int cols, double* result);

#ifdef __cplusplus
}
#endif

#endif // MATVEC_WRAPPER_H