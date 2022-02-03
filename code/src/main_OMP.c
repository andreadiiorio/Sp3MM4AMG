#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "sparseUtilsMulti.h"
#include "SpMMMulti.h"

#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"

#include "sparseMatrix.h"
#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif

//inline funcs
CHUNKS_DISTR    chunksFair,chunksFairFolded,chunksNOOP;
spmat*  allocSpMatrix(ulong rows, ulong cols);
int     allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
spmat*  initSpMatrixSpGEMM(spmat* A, spmat* B);
void    freeSpmatInternal(spmat* mat);
void    freeSpmat(spmat* mat);

CHUNKS_DISTR_INTERF chunkDistrbFunc=&chunksFairFolded;


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
char TRGT_IMPL_START_IDX = 0; //multi implementation switch TODO C


#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1}," \
    "in MatrixMarket_sparse_matrix_COO[compressed], [COMPUTE/PARTITION_MODE: "_ROWS","_SORTED_ROWS","_TILES" ("_ROWS")]\n"

static CONFIG Conf = {
    .gridRows  = 8,
    .gridCols  = 8,
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


    SP3GEMM_INTERF computeFunc = &sp3gemmRowByRowPair_0;
    SPGEMM_INTERF  spgemm = &spgemmRowByRow2DBlocksAllocated_0;
    if (TRGT_IMPL_START_IDX){   //1based indexing implementation
        computeFunc = &sp3gemmRowByRowPair_1;
        spgemm      = &spgemmRowByRow2DBlocksAllocated_1;
    }
    /*TODO COMPREENSIVE UPDATE IN CMODE ... FOCUS ON TEST FILE
     **TODO check on TRGT_IMPL_START_IDX for each case to pick the trgt implentation
    switch (cmode){
        case ROWS:          computeFunc=&sp3gemmGustavsonParallelPair;break;
        //TODO OTHERS
        case TILES:         printf("TODO 2D BLOCKS");break;
    }*/
    spmat *R = NULL, *AC = NULL, *P = NULL, *out = NULL;
    //extra configuration
    int maxThreads = omp_get_max_threads();
    Conf.threadNum = (uint) maxThreads;
    /*
     * get exported schedule configuration, 
     * if chunkSize == 1 set a chunk division function before omp for
     */
    int schedKind_chunk_monotonic[3];
    ompGetRuntimeSchedule(schedKind_chunk_monotonic);
    Conf.chunkDistrbFunc = chunksNOOP; 
    if (schedKind_chunk_monotonic[1] == 1)  Conf.chunkDistrbFunc = chunkDistrbFunc;
    if (!getConfig(&Conf)){
        VERBOSE printf("configuration changed from env");
    }
    ////parse sparse matrixes 
    char* trgtMatrix = TMP_EXTRACTED_MARTIX;
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
    end = omp_get_wtime();elapsed = end-start;
    VERBOSE{printf("preparing time: %le\n",elapsed);print3SPGEMMCore(R,AC,P,&Conf);}
    if (!(out = computeFunc(R,AC,P,&Conf,spgemm))){
        ERRPRINT("compute function selected failed...\n");
        goto _free;
    }

    ret = EXIT_SUCCESS;
    DEBUGPRINT {printf("sparse matrix: AC_i\n");printSparseMatrix(out,TRUE);}
    _free:
    if (R)          freeSpmat(R);
    if (AC)         freeSpmat(AC);
    if (P)          freeSpmat(P);
    if (out)        freeSpmat(out);
    return ret;
}
