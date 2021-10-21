#ifndef _SPGEMM
#define _SPGEMM

#include "macros.h"
#include "sparseMatrix.h"




//compute function interface and its pointer definitions
typedef spmat* ( SPGEMM        )  (spmat*,spmat*,CONFIG*);
typedef spmat* (*SPGEMM_INTERF )  (spmat*,spmat*,CONFIG*);
typedef spmat* ( SP3GEMM       )  (spmat*,spmat*,spmat*,CONFIG*);
typedef spmat* (*SP3GEMM_INTERF)  (spmat*,spmat*,spmat*,CONFIG*);

//aux struct for sparse vector-scalar product accumualtion
typedef struct{
    double* v;          //aux accumulating dense vector
    uint    vLen;       //size of the aux dense vector //TODO USELESS?
    uint*   nnzIdx;     //v, nonzero accumulated values
    uint    nnzIdxLast; //last appended non zero index
} THREAD_AUX_VECT;

///SP3GEMM FUNCTIONS
/*
 *  triple matrix multiplication among @R * @AC * @P using gustavson parallel implementation
 *  implmented as a pair of subsequent spgemm operations
 *  if @conf->spgemm != NULL, it will be used as spgemm function, otherwise euristics will be 
 *  used to decide wich implementation to use
 */
spmat* sp3gemmGustavsonParallel(spmat* R,spmat* AC,spmat* P,CONFIG* conf);

///SUB FUNCTIONS
///SPGEMM FUNCTIONS
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson 
 * with partitioning of @A in @conf->gridRows blocks of rows  
 * used aux dense vector @_auxDense, long @_auxDenseLen. preallocd outside
 * return resulting product matrix
 */
spmat* spgemmGustavsonRowBlocks(spmat* A,spmat* B, CONFIG* conf);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, accessed by aux offsets
 */
spmat* spgemmGustavson2DBlocks(spmat* A,spmat* B, CONFIG* conf);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, ALLOCATED as CSR submatrixes
 */
spmat* spgemmGustavson2DBlocksAllocated(spmat* A,spmat* B, CONFIG* conf);

///HERE array of spgemm function pntr usable, NULL terminated
//#pragma message "C Preprocessor got here!"
static const SPGEMM_INTERF  SpgemmFuncs[] = {
    &spgemmGustavsonRowBlocks,
    &spgemmGustavson2DBlocks,
    &spgemmGustavson2DBlocksAllocated,
    NULL,
};
#endif
