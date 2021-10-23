//sparse matrix def & aux
//TODO adapt to work on both CUDA->ELL and std CSR
#ifndef SPARSEMATRIX
#define SPARSEMATRIX 

#include "macros.h"

typedef struct{
    uint NZ,M,N;
    uint* JA;
    //CSR SPECIFIC
#ifdef ROWLENS
    uint* RL;   //row lengths
#endif
    uint* IRP;
    //CUDA SPECIFIC
    //uint MAXNZ;

    double *AS; 
} spmat; //describe a sparse matrix

////portion of sparse matrix definition for parallel SPGEMM
//CSR   TODO -> macro-generalize for both
typedef struct{
    uint    r;     //row index in the original matrix
    uint    len;   //rowLen
    double* AS;    //row nnz    values
    uint*   JA;    //row nnz    colIndexes
} SPMATROW; 

/*
 * return !0 if col @j idx is in row @i of sparse mat @smat
 */
inline int IS_NNZ(spmat* smat,uint i,uint j){
    int out = 0;
    for (uint x=smat->IRP[i]; x<smat->IRP[i+1] && !out; x++){
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
inline void freeSpRow(SPMATROW* r){ 
    if(r->AS)   free(r->AS);
    if(r->JA)   free(r->JA);
}

//alloc&init internal structures only dependent of dimensions @rows,@cols
inline int allocSpMatrixInternal(uint rows, uint cols, spmat* mat){
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
inline spmat* allocSpMatrix(uint rows, uint cols){

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

//print useful information about 3SPGEMM about to compute
void print3SPGEMMCore(spmat* R,spmat* AC,spmat* P,CONFIG* conf);
#endif
