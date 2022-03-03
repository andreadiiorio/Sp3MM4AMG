//Generic implementations header

#ifndef SPMM_GENERIC_H
#define SPMM_GENERIC_H
///commong funtion -> single implementation

#include "macros.h"
#include "sparseMatrix.h"

///aux structures
//hold SPMM result over a unpartitionated space among threads-row[s' blocks]
typedef struct{
    //space to hold SPMM output
    ulong*  JA;
    double* AS;
    ulong   size;			//num of entries allocated -> only dbg checks
    ulong   lastAssigned;	//last JA&AS assigned index to an accumulator(atom)
    SPACC*  accs;			//SPARSIFIED ACC POINTERS
} SPMM_ACC; //accumulator for SPMM
///compute function interface and its pointer definitions
typedef spmat* ( SPMM        )  (spmat*,spmat*,CONFIG*);
typedef spmat* (*SPMM_INTERF )  (spmat*,spmat*,CONFIG*);
typedef spmat* ( SP3MM       )  (spmat*,spmat*,spmat*,CONFIG*,SPMM_INTERF);
typedef spmat* (*SP3MM_INTERF)  (spmat*,spmat*,spmat*,CONFIG*,SPMM_INTERF);

//aux struct for sparse vector-scalar product accumualtion
typedef struct{
    double* v;          //aux accumulating dense vector
    ulong    vLen;       //size of the aux dense vector //TODO USELESS?
    ulong*   nnzIdx;     //v, nonzero accumulated values
    ulong    nnzIdxLast; //last appended non zero index
} THREAD_AUX_VECT;

#endif //SPMM_GENERIC_H  	////-> end multiImplementation common part

#ifndef OFF_F
    //#pragma message("generic implementation requires OFF_F defined")
    #error generic implementation requires OFF_F defined
#endif

///SP3MM FUNCTIONS
/*
 *  triple matrix multiplication among @R * @AC * @P using gustavson parallel implementation
 *  implmented as a pair of subsequent spmm operations
 *  if @conf->spmm != NULL, it will be used as spmm function, otherwise euristics will be 
 *  used to decide wich implementation to use
 */
SP3MM CAT(sp3mmRowByRowPair_,OFF_F);

/*
 * row-by-row-by-row implementation: forwarding @R*@AC rth row to P for row-by-row
 * accumulation in preallocated space, TODO exactly determined
 * basic parallelization: 1thread per @R's rows that will also forward the result to P
 */
SP3MM CAT(sp3mmRowByRowMerged_,OFF_F);

///SUB FUNCTIONS
///SPMM FUNCTIONS
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson row-by-row
 * formulation using an aux dense vector @_auxDense
 * return resulting product matrix
 */
SPMM CAT(spmmRowByRow_,OFF_F);
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson 
 * with partitioning of @A in @conf->gridRows blocks of rows  
 * return resulting product matrix
 */
SPMM CAT(spmmRowByRow1DBlocks_,OFF_F);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, accessed by aux offsets
 */
SPMM CAT(spmmRowByRow2DBlocks_,OFF_F);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, ALLOCATED as CSR submatrixes
 */
SPMM CAT(spmmRowByRow2DBlocksAllocated_,OFF_F);

///implementation wrappers as static array of function pointers
//sp3mm as pair of spmm
static SPMM_INTERF  CAT(SpmmFuncs_,OFF_F)[] = {
    & CAT(spmmRowByRow_,OFF_F),
    & CAT(spmmRowByRow1DBlocks_,OFF_F),
    & CAT(spmmRowByRow2DBlocks_,OFF_F),
    & CAT(spmmRowByRow2DBlocksAllocated_,OFF_F)
};
//sp3mm as pair of spmm
static SP3MM_INTERF CAT(Sp3mmFuncs_,OFF_F)[] = {
    & CAT(sp3mmRowByRowMerged_,OFF_F)
};
