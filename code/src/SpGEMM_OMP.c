#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

/*
 * basic dual sparse matrix multiplication
 * trivial paralelizzation of Gustavson over rows
 *
 * output matrix will have aux strucutes alloccd here
 */
static int _spgemmRowsBasicDual(spmat* A,spmat* B, spmat* out){
    //init out matrix
    out -> M = A -> M;
    out -> N = B -> N;
    if (!(mat->IRP = calloc(mat->M+1,sizeof(*(mat->IRP))))){
        ERRPRINT("IRP calloc err\n");
        goto err;
    }
#if ROWLENS
    if (!(mat->RL = calloc(mat->M,sizeof(*(mat->RL))))){
        ERRPRINT("IRP calloc err\n");
        goto err;
    }
#endif

}
int spgemmRowsBasic(spmat* R,spmat* AC,spmat* P,CONFIG*, spmat* out){
    int out = EXIT_FAILURE;
    
    double* acc=NULL;
    double end,start;
    start = omp_get_wtime();

     
    end = omp_get_wtime();
    VERBOSE printf("spgemmRowsBasic with %u x %u CSR sp.Mat, elapsed %lf\n",
        mat->M,mat->N,end-start);
    out = EXIT_SUCCESS;
    return out;
}
