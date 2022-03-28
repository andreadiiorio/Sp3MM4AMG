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
#include "sparseMatrix.h"
#include "config.h"

#ifndef OFF_F
	#pragma error OFF_F required
#endif
//////////////////////////////// CSR SPECIFIC /////////////////////////////////
///SPARSE MATRIX PARTITIONING
/*
 * partition CSR sparse matrix @A in @gridCols columns partitions 
 * returning an offsets matrix out[i][j] = start of jth colPartition of row i
 * subdivide @A columns in uniform cols ranges in the output 
 */
idx_t* CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(spmat* A,uint gridCols);

/*
 * partition CSR sparse matrix @A in @gridCols columns partitions as 
 * indipended and allocated sparse matrixes and return them
 * subdivide @A columns in uniform cols ranges in the output 
 */
spmat* CAT(colsPartitioningUnifRanges_,OFF_F)(spmat* A,uint gridCols);
//same as above but with (aux) use of offsets partitoning (also returned if colOffsets!=NULL
spmat* CAT(colsPartitioningUnifRangesOffsetsAux_,OFF_F)(spmat* A,uint gridCols,idx_t** colPartsOffsets);
///////////////////////////////////////////////////////////////////////////////

///Single implementations headers
#ifndef SPARSEUTILS_H_COMMON_IDX_IMPLS
#define SPARSEUTILS_H_COMMON_IDX_IMPLS

//shift every index about the sparse data for use the matric in a fortran application
inline void C_FortranShiftIdxs(spmat* m){
	for(ulong r=0; r<m->M+1; m -> IRP[r]++,r++);
	for(ulong i=0; i<m->NZ;  m -> JA[i]++, i++);
}
//shift every index about the sparse data for use the matric in a C application
inline void Fortran_C_ShiftIdxs(spmat* m){	//TODO DBG ONLY and compleatness
	for(ulong r=0; r<m->M+1; m -> IRP[r]--,r++);
	for(ulong i=0; i<m->NZ;  m -> JA[i]--, i++);
}

/*
 * check SpMM resulting matrix @AB = A * B nnz distribution in rows
 * with the preallocated, forecasted size in @forecastedSizes 
 * in @forecastedSizes there's for each row -> forecasted size 
 * and in the last entry the cumulative of the whole matrix
 */
void checkOverallocPercent(ulong* forecastedSizes,spmat* AB);
//same as above but with 2D partitioning - CSR col partitioning
void checkOverallocRowPartsPercent(ulong* forecastedSizes,spmat* AB,
  idx_t gridCols,idx_t* bColOffsets);
/*  
    check if sparse matrixes A<->B differ up to 
    DOUBLE_DIFF_THREASH per element
*/
int spmatDiff(spmat* A, spmat* B);
////dyn alloc of spMM output matrix
/*
///size prediction of AB = @A * @B
inline ulong SpMMPreAlloc(spmat* A,spmat* B){
    //TODO BETTER PREALLOC HEURISTICS HERE 
    return MAX(A->NZ,B->NZ);
}
//init a sparse matrix AB=@A * @B with a initial allocated space by an euristic
inline spmat* initSpMatrixSpMM(spmat* A, spmat* B){
    spmat* out;
    if (!(out = allocSpMatrix(A->M,B->N)))  return NULL;
    out -> NZ = SpMMPreAlloc(A,B);
    if (!(out->AS = malloc(out->NZ*sizeof(*(out->AS))))){
        ERRPRINT("initSpMatrix: out->AS malloc errd\n");
        free(out);
        return NULL;
    }
    if (!(out->JA = malloc(out->NZ*sizeof(*(out->JA))))){
        ERRPRINT("initSpMatrix: out->JA malloc errd\n");
        freeSpmat(out);
        return NULL;
    }
    return out;
}

#define REALLOC_FACTOR  1.5
//realloc sparse matrix NZ arrays
inline int reallocSpMatrix(spmat* mat,ulong newSize){
    mat->NZ *= newSize;
    void* tmp;
    if (!(tmp = realloc(mat->AS,mat->NZ * sizeof(*(mat->AS))))){
        ERRPRINT("reallocSpMatrix:  realloc AS errd\n");
        return EXIT_FAILURE;
    }
    mat->AS = tmp;
    if (!(tmp = realloc(mat->JA,mat->NZ * sizeof(*(mat->JA))))){
        ERRPRINT("reallocSpMatrix:  realloc JA errd\n");
        return EXIT_FAILURE;
    }
    mat->JA = tmp;
    return EXIT_SUCCESS;
}
*/
////MISC
//print useful information about 3SPMM about to compute
void print3SPMMCore(spmat* R,spmat* AC,spmat* P,CONFIG* conf);
void printSparseMatrix(spmat* sparseMat,char justNZMarkers);
/*convert @sparseMat sparse matrix in dense matrix returned*/
double* CSRToDense(spmat* sparseMat);

#endif //SPARSEUTILS_H_COMMON_IDX_IMPLS 
