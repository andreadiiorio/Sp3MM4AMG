//sparse matrix def & aux
#ifndef SPARSEMATRIX
#define SPARSEMATRIX 

#include <stdlib.h>

#include "macros.h"
#include "config.h"

typedef struct{
    ulong NZ,M,N;
    double *AS; 
    ulong* JA;
    //CSR SPECIFIC
    ulong* IRP;
#ifdef ROWLENS
    ulong* RL;   //row lengths
#endif
    //CUDA SPECIFIC
    ulong MAX_ROW_NZ;

} spmat; //describe a sparse matrix

////Sparse vector accumulator -- corresponding to a matrix portion
typedef struct{
    //ulong    r;     //row index in the corresponding matrix
    //ulong    c;     //col index in the corresponding matrix
    idx_t   len;   //rowLen
    double* AS;    //row nnz    values
    idx_t*  JA;    //row nnz    colIndexes
} SPACC; 


/*
 * ARRAY BISECTION - RECURSIVE VERSION
 * TODO ASSERT LEN>0 ommitted
 */
inline int BISECT_ARRAY(ulong target, ulong* arr, ulong len){
    //if (len == 0)              return FALSE;
    if (len <= 1)              return *arr == target; 
    ulong middleIdx = (len-1) / 2;  //len=5-->2, len=4-->1
    ulong middle    = arr[ middleIdx ];
    if      (target == middle)  return TRUE;
    else if (target <  middle)  return BISECT_ARRAY(target,arr,middleIdx); 
    else    return BISECT_ARRAY(target,arr+middleIdx+1,middleIdx + (len-1)%2);
}

/*
 * return !0 if col @j idx is in row @i of sparse mat @smat
 * bisection used --> O(log_2(ROWLENGHT))
 */
inline int IS_NNZ(spmat* smat,ulong i,ulong j){
    ulong rStart = smat->IRP[i];
    ulong rLen   = smat->IRP[i+1] - rStart;
    if (!rLen)  return FALSE;
    return BISECT_ARRAY(j,smat->JA + rStart,rLen);
}
inline int IS_NNZ_linear(spmat* smat,ulong i,ulong j){    //linear -> O(ROWLENGHT)
    int out = 0;
    for (ulong x=smat->IRP[i]; x<smat->IRP[i+1] && !out; x++){
        out = (j == smat->JA[x]); 
    } 
    return out;
}
////aux functions
//free sparse matrix
inline void freeSpmatInternal(spmat* mat){
    if(mat->AS)    free(mat->AS);  
    if(mat->JA)    free(mat->JA);  
    if(mat->IRP)   free(mat->IRP);  
#ifdef ROWLENS
    if(mat->RL)    free(mat->RL);
#endif 
}
inline void freeSpmat(spmat* mat){
    freeSpmatInternal(mat);
    free(mat);
}

//free max aux structs not NULL pointed
inline void freeSpAcc(SPACC* r){ 
    if(r->AS)   free(r->AS);
    if(r->JA)   free(r->JA);
}
////alloc&init functions
//alloc&init internal structures only dependent of dimensions @rows,@cols
inline int allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat){
    mat -> M = rows;
    mat -> N = cols;
    if (!(mat->IRP=calloc(mat->M+1,sizeof(*(mat->IRP))))){ //calloc only for 0th
        ERRPRINT("IRP calloc err\n");
        freeSpmatInternal(mat);
        return EXIT_FAILURE;
    }
#ifdef ROWLENS
    if (!(mat->RL = malloc(mat->M*sizeof(*(mat->RL))))){
        ERRPRINT("RL calloc err\n");
        freeSpmatInternal(mat);
        return EXIT_FAILURE;
    }
#endif
    return EXIT_SUCCESS;
}

//alloc a sparse matrix of @rows rows and @cols cols 
inline spmat* allocSpMatrix(ulong rows, ulong cols){

    spmat* mat;
    if (!(mat = calloc(1,sizeof(*mat)))) { 
        ERRPRINT("mat  calloc failed\n");
        return NULL;
    }
    if (allocSpMatrixInternal(rows,cols,mat)){
        free(mat);
        return NULL;
    }
    return mat;
}

#endif
