#ifndef SPGEMM
#define SPGEMM

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
typedef spmat* ( COMPUTEFUNC       ) (spmat*,spmat*,spmat*,CONFIG*);
typedef spmat* (*COMPUTEFUNC_INTERF) (spmat*,spmat*,spmat*,CONFIG*);

//aux struct for sparse vector-scalar product accumualtion
typedef struct{
    double* v;          //aux accumulating dense vector
    uint    vLen;       //size of the aux dense vector //TODO USELESS?
    uint*   nnzIdx;     //v, nonzero accumulated values
    uint    nnzIdxLast; //last appended non zero index
} THREAD_AUX_VECT;


/*
 * scalar-vector multiplication performed over @trgtR row of sparse matrix @mat
 * result accumulated in @auxV, with nonzero values indexes in @auxVNNZeroIdxs
 */
void scVectMul(double scalar,spmat* mat,uint trgtR, THREAD_AUX_VECT aux);

/*
 * sparsify accumulated vector @accB into sparse matrix row @accRow
 */
int sparsifyDenseVect(THREAD_AUX_VECT* accV,SPMATROW* accRow);

/*
 * basic spgemm with row partitioning, 1 row per thread in consecutive order
 * statically assigned to threads
 */
spmat* spgemmRowsBasic(spmat* R,spmat* AC,spmat* P,CONFIG* conf);

///SUB FUNCTIONS
/*
 * basic sparse parallel GEMM implementation 
 * paralelizing Gustavson over @conf->gridRows blocks of rows  
 * used aux dense vector @_auxDense, long @_auxDenseLen. preallocd outside
 * return resulting product matrix
 */
spmat* spgemmRowsGustavsonBasic(spmat* A,spmat* B, CONFIG* conf);
#endif
