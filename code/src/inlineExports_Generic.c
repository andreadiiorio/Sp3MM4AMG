#include "macros.h"

#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

////inline exports
//multi implmentation functions
void CAT(scSparseVectMul_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen, THREAD_AUX_VECT* aux);
void CAT(scSparseVectMulPart_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen,ulong startIdx,THREAD_AUX_VECT* aux);
void CAT(_scRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux);
void CAT(scSparseRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux);
idx_t* CAT(spMMSizeUpperbound_,OFF_F)(spmat* A,spmat* B);
idx_t* CAT(spMMSizeUpperboundColParts_,OFF_F)(spmat* A,spmat* B,ushort gridCols,idx_t* bColPartOffsets);
