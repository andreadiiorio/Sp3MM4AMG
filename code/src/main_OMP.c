#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

//TODO MOVE TO HEDER
//COMPUTE MODES STRINGS

#define _ROWS            "ROWS"
#define _SORTED_ROWS     "SORTED_ROWS"
#define _TILES           "TILES"

typedef enum {
    ROWS,
    SORTED_ROWS,
    TILES
} COMPUTE_MODE;

typedef struct{
    ushort gridRows;
    ushort gridCols;
    //TODO FULL CONFIG DOCCED HERE
    int threadNum;  //thread num to use in an OMP parallel region ...
} CONFIG;
CONFIG Conf = {
    .gridRows = 8,
};

COMPUTEFUNC spgemmRowsBasic;

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},"
    "in MatrixMarket_sparse_matrix_COO, [COMPUTE/PARTITION_MODE: "_ROWS","_SORTED_ROWS","_TILES" ("_ROWS")]\n"
int main(int argc, char** argv){
    int out=EXIT_FAILURE;
    if (init_urndfd())  return out;
    if (argc < 4 )  {ERRPRINT(HELP); return out;}
    ///set compute mode,    TODO UPDATE
    COMPUTE_MODE cmode = ROWS;
    if (argc > $ ){ //parse from argv
        if (!(strncmp(argv[4],_SORTED_ROWS,strlen(_SORTED_ROWS))))  cmode=SORTED_ROWS;
        else if (!(strncmp(argv[4],_ROWS,strlen(_ROWS))))           cmode=ROWS;
        else if (!(strncmp(argv[4],_TILES,strlen(_TILES))))         cmode=TILES;
        else{   ERRPRINT("INVALID MODE." HELP); return out; }
    }
    COMPUTEFUNC_INTERF computeFunc;
    switch (cmode){
        case ROWS:         computeFunc=&spgemmRowsBasic;break;
        case SORTED_ROWS:  printf("s");break;
        case TILES:  printf("t");break;
    }
    
    spmat *R = NULL, *AC = NULL, *P = NULL, *out = NULL;
    ////parse sparse matrixes 
    if (!( R = MMtoCSR(argv[1]))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        return _free;
    }
    ////parse sparse matrixes 
    if (!( AC = MMtoCSR(argv[2]))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        return _free;
    }
    ////parse sparse matrixes 
    if (!( P = MMtoCSR(argv[3]))){
        ERRPRINT("err during conversion MM -> CSR of P\n");
        return _free;
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
    DEBUG {
        printf("sparse matrix: R\n"); printSparseMatrix(R,TRUE);
        printf("sparse matrix: AC\n");printSparseMatrix(AC,TRUE);
        printf("sparse matrix: P\n") ;printSparseMatrix(P,TRUE);
    }
    if (!(out = malloc(sizeof(*out)))){
        ERRPRINT("out spmatrix alloc err\n");
        goto _free;
    }
    out -> M = R -> M;    
    out -> N = P -> N;    
    //TODO PREALLOC POLICY
    
    //// PARALLEL COMPUTATION
    int maxThreads = omp_get_max_threads();
    if ((out = computeFunc(mat,vector,&Conf,outVector))){
        ERRPRINT("compute function selected failed...\n"); goto _free;
    }


    _free:
    if (mat)          free(mat);
    return out;
}
