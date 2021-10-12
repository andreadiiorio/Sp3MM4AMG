#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO sparsifyDenseVect EASY CHECKKING
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif


static inline void scRowMul(double scalar,spmat* mat,uint trgtR, THREAD_AUX_VECT* aux){
    for (uint c=mat->IRP[trgtR],j;  c<mat->IRP[trgtR+1];  c++){//scanning trgtRow
        j = mat->JA[c];
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))    aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += mat->AS[c] * scalar;  //accumulate
    }
}

///AUX
/*
 * alloc threads' aux arrays once and split them in threads' structures
 *so free them once from the first thread struct, with the original pointers returned from the alloc
 */
static THREAD_AUX_VECT* _initAccVectors_monoalloc(uint num,uint size){
    THREAD_AUX_VECT* out    = NULL;
    double* vAll            = NULL;
    uint* vAllNzIdx         = NULL;
    if (!(out = calloc(num,sizeof(*out)))){
        ERRPRINT("_initAccVectors aux struct alloc failed\n");
        return NULL;
    }
    if (!(vAll = calloc(num*size,sizeof(*vAll)))) {
        ERRPRINT("_initAccVectors aux dense vectors alloc failed\n");
        goto err;
    }
    if (!(vAllNzIdx = calloc(num*size,sizeof(*vAllNzIdx)))) {
        ERRPRINT("_initAccVectors aux dense vectors' idx alloc failed\n");
        goto err;
    }
    for (uint i=0; i<num; i++){
        out[i].vLen        = size; //TODO USELESS INFO?
        out[i].v           = vAll      + i*size;  
        out[i].nnzIdx      = vAllNzIdx + i*size;;
        out[i].nnzIdxLast  = 0;
    }
    return out;
    
    err:
    free(out);
    if (vAll)        free(vAll);
    if (vAllNzIdx)   free(vAllNzIdx);
    return NULL;
}
//alloc threads' rows accumulators vectors
static THREAD_AUX_VECT* _initAccVectors(uint num,uint size){
    THREAD_AUX_VECT* out    = NULL;
    if (!(out = calloc(num,sizeof(*out)))){
        ERRPRINT("_initAccVectors aux struct alloc failed\n");
        return NULL;
    }
    for (uint i=0; i<num; i++){
        out[i].vLen = size; //TODO USELESS/skippable INFO?
        if (!(out[i].v = calloc(size,sizeof(*(out[i].v))))) {
            ERRPRINT("_initAccVectors aux dense vector alloc failed\n");
            goto err;
        }
        if (!(out[i].nnzIdx = calloc(size,sizeof(*(out[i].nnzIdx))))) {
            ERRPRINT("_initAccVectors aux dense vector' idx alloc failed\n");
            goto err;
        }
        out[i].nnzIdxLast  = 0;
    }
    return out;
    
    err:
    for (uint i=0; i<num && out ; i++){
        if (out[i].v)       free(out[i].v);
        if (out[i].nnzIdx)  free(out[i].nnzIdx);
    }
    free(out);
    return NULL;
}
static inline void _resetAccVect(THREAD_AUX_VECT* acc){
    memset(acc->v,0,acc->vLen * sizeof(*(acc->v)));
    memset(acc->nnzIdx,0,acc->vLen * sizeof(*(acc->nnzIdx)));
    acc->nnzIdxLast = 0;
}
static void _freeAccVectors(THREAD_AUX_VECT* vectors,uint num){
    for (uint i=0; i<num; i++){
        free(vectors[i].v);
        free(vectors[i].nnzIdx);
    }
    free(vectors);
}

int sparsifyDenseVect(THREAD_AUX_VECT* accV,SPMATROW* accRow){
    uint rowNNZ = accV -> nnzIdxLast;
    //alloc row nnz element space
    accRow -> len = rowNNZ;
    if (!(accRow->AS = malloc(rowNNZ * sizeof(*(accRow->AS))))){
        ERRPRINT("sparsifyDenseVect: accRow->AS malloc error\n");
        goto _err;
    }
    if (!(accRow->JA = malloc(rowNNZ * sizeof(*(accRow->JA))))){
        ERRPRINT("sparsifyDenseVect: accRow->JA malloc error\n");
        goto _err;
    }
    sortuint(accV->nnzIdx,rowNNZ); //sort nnz idx for ordered write
    CONSISTENCY_CHECKS{ //TODO SORTING CHECK -> REMOVABLE TODO 
        for( uint i=0; rowNNZ && i < rowNNZ-1; i++){
          if (accV->nnzIdx[i] > accV->nnzIdx[i+1]){ ERRPRINT("NNSORTED!");goto _err;}
        }
    }
    for (uint i=0,j; i<rowNNZ; i++){
        j = accV -> nnzIdx[i];
        accRow -> JA[i] = j;
        accRow -> AS[i] = accV->v[j]; //copy accumulated nz value
    }
    CONSISTENCY_CHECKS{ //TODO check if nnzIdxs sparsify is good as a full scan
        double* tmpNNZ = calloc(accV->vLen, sizeof(*tmpNNZ)); //TODO OVERFLOW POSSIBLE
        if (!tmpNNZ){ERRPRINT("sparsify check tmpNNZ malloc err\n");goto _err;}
        uint ii=0;
        for (uint i=0; i< accV->vLen; i++){
            if ( accV->v[i] )   tmpNNZ[ii++] = accV->v[i];
        }
        if (ii != rowNNZ){
            ERRPRINT("quick sparsify nnz num wrong!\n");
            goto _err;
        }
        //if (memcmp(tmpNNZ,accRow->AS,rowNNZ*sizeof(*tmpNNZ)))
        if (doubleVectorsDiff(tmpNNZ,accRow->AS,rowNNZ))
            {ERRPRINT("quick sparsify check err\n");free(tmpNNZ);goto _err;}
        free(tmpNNZ);
    }
    return EXIT_SUCCESS;
    _err:
    //if(accRow->AS)  free(accRow->AS); //TODO free anyway later
    //if(accRow->JA)  free(accRow->JA);
    return EXIT_FAILURE;    
}

/*
 * merge @mat->M sparse rows @rows in sparse matrix @mat
 * EXPECTED @rows to be sorted in accord to trgt matrix row index @r
 * alloc array to hold non zero values into @mat
 */
int mergeRows(SPMATROW* rows,spmat* mat){
    uint nzNum=0;
    //alloc nnz holder arrays
    for (uint r=0;   r<mat->M;   ++r){
        nzNum += rows[r].len;
        mat->IRP[r+1] = nzNum;
#ifdef ROWLENS
        mat->RL[r]  = rows[r].len
#endif
    }
    mat->NZ = nzNum;
    if (!(mat->AS = malloc(nzNum * sizeof(*(mat->AS))))){
        ERRPRINT("merged sparse matrix AS alloc errd\n");
        return EXIT_FAILURE;
    }  
    if (!(mat->JA = malloc(nzNum * sizeof(*(mat->JA))))){
        ERRPRINT("merged sparse matrix JA alloc errd\n");
        return EXIT_FAILURE;
    }
    //popolate with rows nnz values and indexes
    for (uint r=0; r<mat->M; r++){
        memcpy(mat->AS + mat->IRP[r], rows[r].AS,rows[r].len*sizeof(*(mat->AS)));
        memcpy(mat->JA + mat->IRP[r], rows[r].JA,rows[r].len*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO written nnz check manually
        for (uint r=0,i=0; r<mat->M; r++){
            if (i != mat->IRP[r])
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (uint j=0; j<rows[r].len; j++,i++){
                if (mat->AS[i]!= rows[r].AS[j]){
                    ERRPRINT("MERGE ROW ERR AS\n"); return -1;}
                if (mat->JA[i]   != rows[r].JA[j]){
                    ERRPRINT("MERGE ROW ERR JA\n"); return -1;}
            }
        }
    }
        
    return EXIT_SUCCESS;
} 
    
spmat* spgemmRowsGustavsonBasic(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm rowBlocks\tA:%ux%u * B:%ux%u\n",A->M,A->N,B->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVect = NULL;
    SPMATROW*       accRows  = NULL;
    ///init AB matrix
    spmat* AB;
    if (!(AB = calloc(1,sizeof(*AB)))){
        ERRPRINT("spgemmRowsGustavsonBasic, AB  calloc failed\n");
        return NULL;
    }
    AB -> M = A -> M;
    AB -> N = B -> N;
    //TODO CALLOC NEED CHECK
    if (!(AB->IRP = calloc(AB->M+1,sizeof(*(AB->IRP))))){
        ERRPRINT("IRP calloc err\n");
        goto _err;
    }
#ifdef ROWLENS
    if (!(AB->RL = malloc(AB->M*sizeof(*(AB->RL))))){
        ERRPRINT("RL calloc err\n");
        goto _err;
    }
#endif
    //perform Gustavson over rows blocks -> M / @conf->gridRows
    uint rowBlock = AB->M/conf->gridRows, rowBlockRem = AB->M%conf->gridRows;
    ///aux structures alloc
    if (!(accVect = _initAccVectors(conf->gridRows,AB->N))){
        ERRPRINT("accVect init failed\n");
        goto _err;
    }
    if (!(accRows = calloc(AB->M , sizeof(*accRows)))) {
        ERRPRINT("accRows malloc failed\n");
        goto _err;
    }
    //init sparse row accumulators row indexes //TODO usefull in balance - reorder case
    for (uint r=0; r<AB->M; r++)    accRows[r].r = r;
    

    //TODO OMP
    for (uint b=0,startRow=0,block=rowBlock+(rowBlockRem?1:0); 
         b < conf->gridRows;    
         startRow += block, block=rowBlock+(++b <rowBlockRem?1:0))//dist.unif.
    {
        DEBUG   printf("block %u\t%u:%u(%u)\n",b,startRow,startRow+block-1,block);
        //row-by-row formulation in the given row block
        for (uint r=startRow;  r<startRow+block;  r++){
            //iterate over nz col index j inside current row r
            for (uint j=A->IRP[r]; j<A->IRP[r+1]; j++) //row-by-row formul. accumulation
                scRowMul(A->AS[j], B, A->JA[j], accVect+b);
            //trasform accumulated dense vector to CSR with using the aux idxs
            if (sparsifyDenseVect(accVect+b,accRows + r)){
                fprintf(stderr,"sparsify failed ar row %u.\t aborting.\n",r);
                //TODO ABORT OPENMP COMPUTATION
                goto _err;
            }
            _resetAccVect(accVect+b);   //rezero for the next A row
        }
    }
    ///merge sparse row computed before
    if (mergeRows(accRows,AB))    goto _err;


    goto _free;
    
    _err:
    freeSpmat(AB);
    AB=NULL;    //nothing'll be returned
    _free:
    for (uint r=0; r<AB->M; r++)    freeSpRow(accRows + r);
    free(accRows);
    _freeAccVectors(accVect,conf->gridRows);

    return AB;
}

spmat* spgemmRowsBasic(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    
    double end,start;
    start = omp_get_wtime();
   
    //alloc dense aux vector, reusable over 3 product 
    uint auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    //TODO PENSA BENE SE AGGIUNGERE RIUSO DI VETTORE DENSO NELLE 2 ALLOCAZIONE CON LA DIMENSIONE MASSIMA TRA LE 2 MATRICI.N........
    //TODO PRODOTTO IN 2
    
    spmat *RAC = NULL, *out = NULL;
    if (!(RAC = spgemmRowsGustavsonBasic(R,AC,conf)))    goto _free;
#ifdef DEBUG_TEST_CBLAS
    DEBUG{ if(denseGEMMCheckCBLAS(R,AC,RAC))             goto _free; }
#endif
    if (!(out = spgemmRowsGustavsonBasic(RAC,P,conf)))   goto _free;
#ifdef DEBUG_TEST_CBLAS
    DEBUG{ if(denseGEMMCheckCBLAS(RAC,P,out))            goto _free; }
#endif
     
    end = omp_get_wtime();
    VERBOSE 
    printf("spgemmRowsBasic 3 product of R:%ux%u AC:%ux%u P:%ux%u CSR sp.Mat, "
        "elapsed %lf\n",    R->M,R->N,AC->M,AC->N,P->M,P->N,end-start);

    _free:
    if (RAC)    freeSpmat(RAC);

    return out;

}
