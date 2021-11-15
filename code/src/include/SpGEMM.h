#ifndef _SPGEMM
#define _SPGEMM

#include "macros.h"
#include "sparseMatrix.h"

///OUTPUT SIZE PREDICTION
/*
 * return for each spGEMM output matrix row -> upper bound size
 * also an extra position at the end for the cumulative total size of the 
 * output matrix AB = A*B
 * O(A.NZ)
 */
ulong* spGEMMSizeUpperbound(spmat* A,spmat* B);
typedef struct{
    //space to hold SPGEMM output
    ulong    size;
    ulong*   JA;
    double* AS;
    ulong    lastAssigned;   //last JA&AS assigned index to an accumulator //TODO OMP ATOMIC
    SPACC*  accs;
} SPGEMM_ACC; //accumulator for SPGEMM
/* 
 * init an spgemm op accumulator, that whill hold @entriesNum nnz entries, pointed by
 * pointed by @accumulatorsNum sparse vector accumulators
 */
SPGEMM_ACC* initSpGEMMAcc(ulong entriesNum, ulong accumulatorsNum);

//compute function interface and its pointer definitions
typedef spmat* ( SPGEMM        )  (spmat*,spmat*,CONFIG*);
typedef spmat* (*SPGEMM_INTERF )  (spmat*,spmat*,CONFIG*);
typedef spmat* ( SP3GEMM       )  (spmat*,spmat*,spmat*,CONFIG*);
typedef spmat* (*SP3GEMM_INTERF)  (spmat*,spmat*,spmat*,CONFIG*);

//aux struct for sparse vector-scalar product accumualtion
typedef struct{
    double* v;          //aux accumulating dense vector
    ulong    vLen;       //size of the aux dense vector //TODO USELESS?
    ulong*   nnzIdx;     //v, nonzero accumulated values
    ulong    nnzIdxLast; //last appended non zero index
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
 * sparse parallel implementation of @A * @B parallelizing Gustavson row-by-row
 * formulation using an aux dense vector @_auxDense
 * return resulting product matrix
 */
spmat* spgemmGustavsonRows(spmat* A,spmat* B, CONFIG* conf);
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson 
 * with partitioning of @A in @conf->gridRows blocks of rows  
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
    &spgemmGustavsonRows,
    &spgemmGustavsonRowBlocks,
    &spgemmGustavson2DBlocks,
    &spgemmGustavson2DBlocksAllocated,
    NULL,
};
#endif
