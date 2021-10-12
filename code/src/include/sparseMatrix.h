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

//free sparse row
inline void freeSpmat(spmat* mat){
    if(mat->AS)    free(mat->AS);  
    if(mat->JA)    free(mat->JA);  
    if(mat->IRP)   free(mat->IRP);  
#ifdef ROWLENS
    if(mat->RL)    free(mat->RL);
#endif 
    free(mat);
}

//free max aux structs not NULL pointed
inline void freeSpRow(SPMATROW* r){ 
    free(r->AS);
    free(r->JA);
}

#endif
