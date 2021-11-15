#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"

#include "sparseMatrix.h"
///inline redef here
//void freeSpmat(spmat* mat);

#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif

SP3GEMM sp3gemmGustavsonParallel;

//COMPUTE MODES 
typedef enum {
    ROWS,
    SORTED_ROWS,
    TILES
} COMPUTE_MODE;
#define _ROWS            "ROWS"
#define _SORTED_ROWS     "SORTED_ROWS"
#define _TILES           "TILES"


//global vars	-	audit
//double Start;

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1}," \
    "in MatrixMarket_sparse_matrix_COO[compressed], [COMPUTE/PARTITION_MODE: "_ROWS","_SORTED_ROWS","_TILES" ("_ROWS")]\n"

static CONFIG Conf = {
    .gridRows  = 8,
    .gridCols  = 8,
    .spgemmFunc=NULL
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
    double end,start,elapsed,flops;
    start = omp_get_wtime();


    SP3GEMM_INTERF computeFunc;
    switch (cmode){
        case ROWS:          computeFunc=&sp3gemmGustavsonParallel;break;
        //TODO OTHERS
        case SORTED_ROWS:   printf("SORTED_ROWS TODO");break;
        case TILES:         printf("TODO 2D BLOCKS");break;
    }
    
    spmat *R = NULL, *AC = NULL, *P = NULL, *out = NULL;
    char* trgtMatrix;
    ////parse sparse matrixes 
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[1],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[1];
    if (!( R = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        goto _free;
    }
    ////parse sparse matrixes 
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[2],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[2];
    if (!( AC = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        goto _free;
    }
    ////parse sparse matrixes 
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[3],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[3];
    if (!( P = MMtoCSR(trgtMatrix))){
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
    int maxThreads = omp_get_max_threads();
    Conf.threadNum = (uint) maxThreads;
    end = omp_get_wtime();elapsed = end-start;
    VERBOSE{printf("preparing time: %le\n",elapsed);print3SPGEMMCore(R,AC,P,&Conf);}
    if (!(out = computeFunc(R,AC,P,&Conf))){
        ERRPRINT("compute function selected failed...\n");
        goto _free;
    }

    ret = EXIT_SUCCESS;
    DEBUGPRINT {printf("sparse matrix: AC_i\n");printSparseMatrix(out,TRUE);}
#ifdef DEBUG_TEST_CBLAS
    DEBUG{
        if ((ret = GEMMTripleCheckCBLAS(R,AC,P,out)))
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
