//Get multiple implementation for C-Fortran indexing by re-define & re-include
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO quick hold few CSR cols partition sizes
#include <omp.h>
//UB version deps
#include "Sp3MM_CSR_OMP_Multi.h"
#include "SpMMUtilsMulti.h"
#include "sparseUtilsMulti.h"
#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"
//inline export - single implmentation functions
idx_t reductionMaxSeq(idx_t* arr,idx_t arrLen);
SPMM_ACC* initSpMMAcc(ulong entriesNum, ulong accumulatorsNum);
void freeSpMMAcc(SPMM_ACC* acc);
void sparsifyDenseVect(SPMM_ACC* acc,THREAD_AUX_VECT* accV,SPACC* accSparse, ulong startColAcc);
int mergeRowsPartitions(SPACC* rowsParts,spmat* mat,CONFIG* conf);
int mergeRows(SPACC* rows,spmat* mat);
THREAD_AUX_VECT* _initAccVectors_monoalloc(ulong num,ulong size); //TODO PERF WITH NEXT
int _allocAuxVect(THREAD_AUX_VECT* v,ulong size);
void _resetAccVect(THREAD_AUX_VECT* acc);
void _freeAccVectorsChecks(THREAD_AUX_VECT* vectors,ulong num); 
void freeAccVectors(THREAD_AUX_VECT* vectors,ulong num);
//void C_FortranShiftIdxs(spmat* outMat);
//void Fortran_C_ShiftIdxs(spmat* m);


//Symb version deps
#include "Sp3MM_CSR_OMP_Multi.h"


//global vars	->	audit
double Start,End,Elapsed,ElapsedInternal;

#define OFF_F 0
#include "inlineExports_Generic.c"
#include "Sp3MM_CSR_OMP_UB_Generic.c"
//#include "Sp3MM_CSR_OMP_Symb_Generic.c"
#undef OFF_F

#define OFF_F 1
#include "inlineExports_Generic.c"
#include "Sp3MM_CSR_OMP_UB_Generic.c"
//#include "Sp3MM_CSR_OMP_Symb_Generic.c"
#undef OFF_F
