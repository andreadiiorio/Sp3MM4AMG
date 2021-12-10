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
typedef spmat* ( SP3GEMM       )  (spmat*,spmat*,spmat*,CONFIG*,SPGEMM_INTERF);
typedef spmat* (*SP3GEMM_INTERF)  (spmat*,spmat*,spmat*,CONFIG*,SPGEMM_INTERF);

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
SP3GEMM sp3gemmRowByRowPair;

/*
 * row-by-row-by-row implementation: forwarding @R*@AC rth row to P for row-by-row
 * accumulation in preallocated space, TODO exactly determined
 * basic parallelization: 1thread per @R's rows that will also forward the result to P
 */
SP3GEMM sp3gemmRowByRowMerged;

///SUB FUNCTIONS
///SPGEMM FUNCTIONS
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson row-by-row
 * formulation using an aux dense vector @_auxDense
 * return resulting product matrix
 */
SPGEMM spgemmRowByRow;
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson 
 * with partitioning of @A in @conf->gridRows blocks of rows  
 * return resulting product matrix
 */
SPGEMM spgemmRowByRow1DBlocks;

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, accessed by aux offsets
 */
SPGEMM spgemmRowByRow2DBlocks;

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, ALLOCATED as CSR submatrixes
 */
SPGEMM spgemmRowByRow2DBlocksAllocated;

///implementation wrappers as static array of function pointers
//sp3gemm as pair of spgemm
static const SPGEMM_INTERF  SpgemmFuncs[] = {
    &spgemmRowByRow,
    &spgemmRowByRow1DBlocks,
    &spgemmRowByRow2DBlocks,
    &spgemmRowByRow2DBlocksAllocated
};
//sp3gemm as pair of spgemm
static const SP3GEMM_INTERF Sp3gemmFuncs[] = {
    &sp3gemmRowByRowMerged,
};
#endif
