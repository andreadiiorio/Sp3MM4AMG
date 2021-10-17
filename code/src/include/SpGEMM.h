#ifndef _SPGEMM
#define _SPGEMM

#include "macros.h"
#include "sparseMatrix.h"
//COMPUTE MODES STRINGS

#define _ROWS            "ROWS"
#define _SORTED_ROWS     "SORTED_ROWS"
#define _TILES           "TILES"

typedef enum {
    ROWS,
    SORTED_ROWS,
    TILES
} COMPUTE_MODE;

typedef struct{
    ushort gridRows;
    ushort gridCols;
    //TODO FULL CONFIG DOCCED HERE
    int threadNum;  //thread num to use in an OMP parallel region ...
} CONFIG;



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


/*
 * basic spgemm with row partitioning, 1 row per thread in consecutive order
 * statically assigned to threads
 */
spmat* sp3gemmGustavsonParallel(spmat* R,spmat* AC,spmat* P,CONFIG* conf);

///SUB FUNCTIONS
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

#endif
