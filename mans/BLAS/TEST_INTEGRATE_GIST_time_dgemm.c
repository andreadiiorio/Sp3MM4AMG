/*
 * enahanced from https://gist.github.com/xianyi/5780018
 * gcc -o TEST_INTEGRATE_GIST_time_dgemm.o -std=c99 TEST_INTEGRATE_GIST_time_dgemm.c build/lib/libopenblas.a -pthread -Wall -Wextra
 * adapted for a regular mat mul
 */
#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"

void openblas_set_num_threads(int num_threads);
extern void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);

typedef unsigned int uint;
static void printMatrix(uint M,uint N,double mat[][N]){
    for(uint i=0; i<M; i++){
        for(uint j=0; j<N; j++)     printf("%1.1le ",mat[i][j]);
        printf("\n");
    }
    printf("\n\n");
}
int main(int argc, char* argv[])
{
	int i;
	printf("test!\n");
	if(argc<4){
		printf("Input Error\n");
		return 1;
	}
 

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	int sizeofa = m * k;
	int sizeofb = k * n;
	int sizeofc = m * n;
	char ta = 'N';
	char tb = 'N';
	double alpha = 1;
	double beta	= 1;

	struct timeval start,finish;
	double duration;

	double* A = malloc(sizeof(*A) * sizeofa);
	double* B = malloc(sizeof(*B) * sizeofb);
	double* C = malloc(sizeof(*C) * sizeofc);

	srand((unsigned)time(NULL));

	for (i=0; i<sizeofa; i++)		A[i] = i%3+1;//(rand()%100)/10.0;
	for (i=0; i<sizeofb; i++)		B[i] = i%3+1;//(rand()%100)/10.0;
	//for (i=0; i<sizeofc; i++)		C[i] = i%3+1;//(rand()%100)/10.0;
	printMatrix(m,k,(double (*)[k]) A); printMatrix(k,n,(double (*)[n])B);
	printf("dgemm_:: m=%d,n=%d,k=%d,alpha=%lf,beta=%lf,sizeofc=%d\n",m,n,k,alpha,beta,sizeofc);
	gettimeofday(&start, NULL);
	////// OPENBLAS LINK
	openblas_set_num_threads(1); ///TODO SERIAL EXECUTION
	dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
	///////////////////
	gettimeofday(&finish, NULL);
    printMatrix(m,n,(double (*)[n])C);    

	duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
	double gflops = 2.0 * m *n*k;
	gflops = gflops/duration*1.0e-6;
	
	printf("%dx%dx%d\t%lf s\t%lf MFLOPS\n", m, n, k, duration, gflops);

	free(A);
	free(B);
	free(C);
	return 0;
}
