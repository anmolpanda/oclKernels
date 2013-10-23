#ifndef C_UTILS_H
#define C_UTILS_H 

#ifdef __cplusplus
extern "C"{
#endif

int getLineNumber(char *fileName);

void read_frames(char *file, float **frames,   int col,   int row);

void init_2d_f(float *frames, int row, int col, float val);

void init_1d_f(float *array, int N, float val);

void init_1d_d(double *array,   int N, double val);

void check_3d_f(float ***array,   int row,   int col,   int N);

void check_2d_f(float *array, int row, int col);

void check_1d_f(float *array, int len);

void rand_1d_f(float *a,   int N);

void rand_2d_f(float **a,   int row,   int col);

void randperm_1d(int *a,   int N);

void transpose(float **in, float **out,   int row,   int col);

void symmetrize_f(float **in,   int N);

void normalise_2d_f(float **x,   int row,   int col);

void normalise_1d_f(float *x,   int len);

void mk_stochastic_2d_f(float **x,   int row,   int col, float **out);

float norm_square(float **U,   int j,   int rows);

void eye_2d_d(double **U,  int row,   int col);

void get2ddiag_d(double **from,double *to,   int row,   int col);

void copy_2d_d(double **a, double **a_pre, int row, int col);

void copy_1d_d(double *from, double *to, int len);

int cmp_2d_f(float *a, float *b, int row, int col);






#ifdef __cplusplus
}
#endif

#endif
