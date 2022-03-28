/*
 * Copyright Andrea Di Iorio 2022
 * This file is part of Sp3MM_for_AlgebraicMultiGrid
 * Sp3MM_for_AlgebraicMultiGrid is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Sp3MM_for_AlgebraicMultiGrid is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Sp3MM_for_AlgebraicMultiGrid.  If not, see <http://www.gnu.org/licenses/>.
 */ 
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
