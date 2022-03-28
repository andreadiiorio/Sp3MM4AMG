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
//inline export - single implmentation functions
void cleanRbNodes(rbRoot* root,rbNode* nodes,idx_t nodesNum);
int rbInsertNewKey(rbRoot *root,rbNode *node, idx_t key);
//void C_FortranShiftIdxs(spmat* outMat);
//void Fortran_C_ShiftIdxs(spmat* m);
ACC_DENSE* _initAccVectors(ulong num,ulong size);
ACC_DENSE* _initAccVectors_monoalloc(ulong num,ulong size); //TODO PERF WITH NEXT
SPMM_ACC* initSpMMAcc(ulong entriesNum, ulong accumulatorsNum);
idx_t reductionMaxSeq(idx_t* arr,idx_t arrLen);
int _allocAccDense(ACC_DENSE* v,ulong size);
int mergeRows(SPACC* rows,spmat* mat);
int mergeRowsPartitions(SPACC* rowsParts,spmat* mat,CONFIG* conf);
void _freeAccsDenseChecks(ACC_DENSE* vectors,ulong num); 
void _resetAccVect(ACC_DENSE* acc);
void _resetIdxMap(SPVECT_IDX_DENSE_MAP* acc);
void assertArrNoRepetitions(idx_t* arrSorted, idx_t arrLen);
void freeAccsDense(ACC_DENSE* vectors,ulong num);
void freeSpMMAcc(SPMM_ACC* acc);
void sparsifyDenseVect(SPMM_ACC* acc,ACC_DENSE* accV,SPACC* accSparse, ulong startColAcc);
