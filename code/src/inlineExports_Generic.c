#include "macros.h"

#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

////inline exports
//multi implmentation functions
void CAT(scSparseVectMul_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen, ACC_DENSE* aux);
void CAT(scSparseVectMulPart_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen,ulong startIdx,ACC_DENSE* aux);
void CAT(_scRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, ACC_DENSE* aux);
void CAT(scSparseRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, ACC_DENSE* aux);
idx_t* CAT(spMMSizeUpperbound_,OFF_F)(spmat* A,spmat* B);
idx_t* CAT(spMMSizeUpperboundColParts_,OFF_F)(spmat* A,spmat* B,ushort gridCols,idx_t* bColPartOffsets);
