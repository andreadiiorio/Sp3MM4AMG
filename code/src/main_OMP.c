#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"

#include "sparseMatrix.h"
extern void freeSpRow(SPMATROW* r);
extern void freeSpmat(spmat* mat);

#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif

COMPUTEFUNC spgemmRowsBasic;

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1}," \
    "in MatrixMarket_sparse_matrix_COO, [COMPUTE/PARTITION_MODE: "_ROWS","_SORTED_ROWS","_TILES" ("_ROWS")]\n"

CONFIG Conf = {
    .gridRows  = 8,
    .threadNum = 8,
};

int main(int argc, char** argv){
    int ret=EXIT_FAILURE;
    if (init_urndfd())  return ret;
    if (argc < 4 )  {ERRPRINT(HELP); return ret;}
    ///set compute mode,    TODO UPDATE
    COMPUTE_MODE cmode = ROWS;
    if (argc > 4 ){ //parse from argv
        if (!(strncmp(argv[4],_SORTED_ROWS,strlen(_SORTED_ROWS))))  cmode=SORTED_ROWS;
        else if (!(strncmp(argv[4],_ROWS,strlen(_ROWS))))           cmode=ROWS;
        else if (!(strncmp(argv[4],_TILES,strlen(_TILES))))         cmode=TILES;
        else{   ERRPRINT("INVALID MODE." HELP); return ret; }
    }
    COMPUTEFUNC_INTERF computeFunc;
    switch (cmode){
        case ROWS:         computeFunc=&spgemmRowsBasic;break;
        //TODO OTHERS
        case SORTED_ROWS:  printf("s");break;
        case TILES:  printf("t");break;
    }
    
    spmat *R = NULL, *AC = NULL, *P = NULL, *out = NULL;
    ////parse sparse matrixes 
    if (!( R = MMtoCSR(argv[1]))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        goto _free;
    }
    ////parse sparse matrixes 
    if (!( AC = MMtoCSR(argv[2]))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        goto _free;
    }
    ////parse sparse matrixes 
    if (!( P = MMtoCSR(argv[3]))){
        ERRPRINT("err during conversion MM -> CSR of P\n");
        goto _free;
    }
    
    CONSISTENCY_CHECKS{
        if (R->N != AC ->M){
            ERRPRINT("invalid sizes in R <-> AC\n");
            goto _free;
        }
        if (AC->N != P ->M){
            ERRPRINT("invalid sizes in AC <-> P\n");
            goto _free;
        }
    }
    DEBUGPRINT {
        printf("sparse matrix: R_i+1\n"); printSparseMatrix(R,TRUE);
        printf("sparse matrix: AC_i\n");  printSparseMatrix(AC,TRUE);
        printf("sparse matrix: P_i+1\n"); printSparseMatrix(P,TRUE);
    }
    
    //// PARALLEL COMPUTATION
    ////TODO int maxThreads = omp_get_max_threads();
    if (!(out = computeFunc(R,AC,P,&Conf))){
        ERRPRINT("compute function selected failed...\n");
        goto _free;
    }

    ret = EXIT_SUCCESS;
    DEBUGPRINT {printf("sparse matrix: AC_i\n");printSparseMatrix(out,TRUE);}
#ifdef DEBUG_TEST_CBLAS
    DEBUG{
        if ((ret = denseGEMMTripleCheckCBLAS(R,AC,P,out)))
            ERRPRINT("LAPACK.CBLAS SERIAL, DENSE GEMM TEST FAILED!!\n");
    }
#endif
    _free:
    if (R)          freeSpmat(R);
    if (AC)         freeSpmat(AC);
    if (P)          freeSpmat(P);
    if (out)        freeSpmat(out);
    return ret;
}
