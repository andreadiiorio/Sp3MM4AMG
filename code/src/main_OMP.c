/*
 *              Sp3MM_for_AlgebraicMultiGrid
 *    (C) Copyright 2021-2022
 *        Andrea Di Iorio      
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions, and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the Sp3MM_for_AlgebraicMultiGrid or the names of its contributors may
 *       not be used to endorse or promote products derived from this
 *       software without specific written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 *  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE Sp3MM_for_AlgebraicMultiGrid GROUP OR ITS CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */ 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "sparseUtilsMulti.h"
#include "Sp3MM_CSR_OMP_Multi.h"

#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"

#include "sparseMatrix.h"
#ifdef DEBUG_TEST_CBLAS
    #include "SpMM_OMP_test.h"
#endif

//inline funcs
CHUNKS_DISTR    chunksFair,chunksFairFolded,chunksNOOP;
spmat*  allocSpMatrix(ulong rows, ulong cols);
int     allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
spmat*  initSpMatrixSpMM(spmat* A, spmat* B);
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


    SP3MM_INTERF computeFunc = &sp3mmRowByRowPair_0;
    SPMM_INTERF  spmm = &spmmRowByRow2DBlocksAllocated_0;
    if (TRGT_IMPL_START_IDX){   //1based indexing implementation
        computeFunc = &sp3mmRowByRowPair_1;
        spmm      	= &spmmRowByRow2DBlocksAllocated_1;
    }
    /*TODO COMPREENSIVE UPDATE IN CMODE ... FOCUS ON TEST FILE
     **TODO check on TRGT_IMPL_START_IDX for each case to pick the trgt implentation
    switch (cmode){
        case ROWS:          computeFunc=&sp3mmGustavsonParallelPair;break;
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
    VERBOSE{printf("preparing time: %le\n",elapsed);print3SPMMCore(R,AC,P,&Conf);}
    if (!(out = computeFunc(R,AC,P,&Conf,spmm))){
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
