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
//Developped by     Andrea Di Iorio - 0277550
#pragma message( "compiling SpMM_CSR_OMP_Generic.c with OFF_F as:" STR(OFF_F) )
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

#ifndef SP3MM_OMP_SYMB
#define SP3MM_OMP_SYMB

//allocCSRSpMatSymbStep aux function for IRP set by symb step output
static inline idx_t _setCSR_IRP_1DPartitioing(spmat* m,idx_t* rowSizes){
	idx_t r,cumulSize;
	for (r=0,cumulSize=0; r<m->M; cumulSize += rowSizes[r++])	m->IRP[r] 	= cumulSize;
	DEBUGCHECKS		assert(cumulSize == rowSizes[m->M]);
	return cumulSize;
}
static inline idx_t _setCSR_IRP_2DPartitioing(spmat* m,idx_t* rowSizes,ushort gridCols){
	idx_t cumulSize=0;
	for (idx_t r=0,cumulSizeOld; r<m->M; r++){
		m->IRP[r] = cumulSize;
		for (ushort gc=0; gc<gridCols; gc++){
			cumulSizeOld = cumulSize;
			cumulSize += rowSizes[ IDX2D(r,gc,gridCols) ];
			//inplace update 2D rowParts len matrix as IRP like	start offsets
			rowSizes[ IDX2D(r,gc,gridCols) ] = cumulSizeOld; 
		}
	}
	DEBUGCHECKS		assert(cumulSize == rowSizes[m->M*gridCols]);
	return cumulSize;
}
/*
 * allocate CSR mat @m, setting up correctly IRP and other buffers allocations
 * if @gridCols == 1: (1D partitioning of @m) -> @rowsSizes will be an array of row lenghts
 * if @gridCols >  1: (2D partitioning of @m) -> @rowsSizes will be a matrix with
 * 		elem i,j = len of j-th colPartition (out of @gridCols) of i-th row
 */
static inline int allocCSRSpMatSymbStep(spmat* m,idx_t* rowSizes,ushort gridCols){
	//setup IRP and get cumul, whole size of @m
	DEBUGCHECKS		assert(gridCols >= 1);
	idx_t cumulSize;
	if (gridCols == 1)	cumulSize = _setCSR_IRP_1DPartitioing(m,rowSizes);
	if (gridCols > 1)	cumulSize = _setCSR_IRP_2DPartitioing(m,rowSizes,gridCols);
	m->IRP[m->M] 	= cumulSize;
	m->NZ			= cumulSize;

	if (!(m->AS = malloc(rowSizes[m->M] * sizeof(*m->AS))) ){
		ERRPRINT("allocCSRSpMatSymbStep m->AS malloc errd\n");
		return EXIT_FAILURE;
	}
	if (!(m->JA = malloc(rowSizes[m->M] * sizeof(*m->JA))) ){
		ERRPRINT("allocCSRSpMatSymbStep m->JA malloc errd\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
#endif
//////////////////// COMPUTE CORE Sp[3]MM SYMB-NUMB PHASE //////////////////////////
////////Sp3MM as 2 x SpMM
///1D
spmat* CAT(spmmRowByRow_SymbNum_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    ACC_DENSE *accVects = NULL,*acc;
    SPMM_ACC* outAccumul=NULL;
    idx_t* rowsSizes = NULL;
    ///init AB matrix with SPMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
	///SYMBOLIC STEP
	if (!(rowsSizes = CAT(SpMM_Symb___,OFF_F) (cfg->symbMMRowImplID, A,B)))	goto _err;
	if (allocCSRSpMatSymbStep(AB,rowsSizes,1))			goto _err;
	
    ///NUMERIC STEP
    ((CHUNKS_DISTR_INTERF)	cfg->chunkDistrbFunc) (AB->M,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(acc)
    for (idx_t r=0;  r<A->M; r++){    //row-by-row formulation
    	acc = accVects + omp_get_thread_num();
		_resetAccVect(acc);   //rezero for the next A row
		//for each A's row nnz, accumulate scalar vector product nnz.val * B[nnz.col]
		/*	direct use of sparse scalar vector multiplication
    	for (idx_t ja=A->IRP[r]-OFF_F,ca,jb,bRowLen;  ja<A->IRP[r+1]-OFF_F;  ja++){
    	    ca 		= A->JA[ja]		- OFF_F;
			jb 		= B->IRP[ca]	- OFF_F;
			bRowLen = B->IRP[ca+1] 	- B->IRP[ca];
    	    CAT(scSparseVectMul_,OFF_F)(A->AS[ja], B->AS+jb,B->JA+jb,bRowLen,acc);
		} */
    	for (ulong ja=A->IRP[r]-OFF_F;	ja<A->IRP[r+1]-OFF_F;	ja++) //row-by-row formul
    	    CAT(scSparseRowMul_,OFF_F)(A->AS[ja], B, A->JA[ja]-OFF_F, acc);
    	//direct sparsify: trasform accumulated dense vector to a CSR row
    	sparsifyDirect(acc,AB,r); //0,NULL);TODO COL PARTITIONING COMMON API
    }
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   checkOverallocPercent(rowsSizes,AB);
    goto _free;

    _err:
    if(AB)  freeSpmat(AB);
    AB=NULL;    //nothing'll be returned
    _free:
    if(rowsSizes)   free(rowsSizes);
    if(accVects)    freeAccsDense(accVects,cfg->threadNum);
    if(outAccumul)  freeSpMMAcc(outAccumul);

    return AB;

}
spmat* CAT(spmmRowByRow1DBlocks_SymbNum_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    ACC_DENSE *accVects = NULL,*acc;
    SPMM_ACC* outAccumul=NULL;
    idx_t* rowsSizes = NULL;
    ///init AB matrix with SPMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
	///SYMBOLIC STEP
	if (!(rowsSizes = CAT(SpMM_Symb___,OFF_F) (cfg->symbMMRowImplID, A,B)))	goto _err;
	if (allocCSRSpMatSymbStep(AB,rowsSizes,1))			goto _err;
	
    ///NUMERIC STEP
    //perform Gustavson over rows blocks -> M / @cfg->gridRows
    ulong rowBlock = AB->M/cfg->gridRows, rowBlockRem = AB->M%cfg->gridRows;
    ((CHUNKS_DISTR_INTERF)	cfg->chunkDistrbFunc) (cfg->gridRows,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    ulong b,startRow,block; //omp for aux vars
    #pragma omp parallel for schedule(runtime) private(acc,startRow,block)
    for (b=0;   b < cfg->gridRows; b++){
        block      = UNIF_REMINDER_DISTRI(b,rowBlock,rowBlockRem);
        startRow   = UNIF_REMINDER_DISTRI_STARTIDX(b,rowBlock,rowBlockRem);
    	for (idx_t r=startRow;  r<startRow+block; r++){
    		acc = accVects + omp_get_thread_num();
			_resetAccVect(acc);   //rezero for the next A row
			//for each A's row nnz, accumulate scalar vector product nnz.val * B[nnz.col]
			/*	direct use of sparse scalar vector multiplication
    		for (idx_t ja=A->IRP[r]-OFF_F,ca,jb,bRowLen;  ja<A->IRP[r+1]-OFF_F;  ja++){
    		    ca 		= A->JA[ja]		- OFF_F;
				jb 		= B->IRP[ca]	- OFF_F;
				bRowLen = B->IRP[ca+1] 	- B->IRP[ca];
    		    CAT(scSparseVectMul_,OFF_F)(A->AS[ja], B->AS+jb,B->JA+jb,bRowLen,acc);
			} */
    		for (ulong ja=A->IRP[r]-OFF_F;	ja<A->IRP[r+1]-OFF_F;	ja++) //row-by-row formul
    		    CAT(scSparseRowMul_,OFF_F)(A->AS[ja], B, A->JA[ja]-OFF_F, acc);
    		//direct sparsify: trasform accumulated dense vector to a CSR row
    		sparsifyDirect(acc,AB,r); //0,NULL);TODO COL PARTITIONING COMMON API
    	}
	}
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   checkOverallocPercent(rowsSizes,AB);
    goto _free;

    _err:
    if(AB)  freeSpmat(AB);
    AB=NULL;    //nothing'll be returned
    _free:
    if(rowsSizes)   free(rowsSizes);
    if(accVects)    freeAccsDense(accVects,cfg->threadNum);
    if(outAccumul)  freeSpMMAcc(outAccumul);

    return AB;

}
spmat* CAT(spmmRowByRow2DBlocks_SymbNum_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){ 
    DEBUG printf("spmm\trowBlocks of A ,\tcolBlocks of B\tM=%luxN=%lu\n",A->M,B->N);
    idx_t* bColOffsets = NULL;   //B group columns starting offset for each row
    ACC_DENSE *accVectors=NULL,*accV;
    SPACC* accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    SPMM_ACC* outAccumul=NULL;
    idx_t    *rowsPartsSizes=NULL, *rowSizes=NULL;	//for rows' cols partition, correct len
    if (!AB)    goto _err;
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, aSubRowsN=A->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////get bColOffsets for B column groups 
    if (!(bColOffsets = CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(B,cfg->gridCols)))
		goto _err;
	///TODO TODO
    _err:
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    free(rowsPartsSizes);
    free(bColOffsets);
    if (accVectors)  freeAccsDense(accVectors,gridSize);
    if (outAccumul)  freeSpMMAcc(outAccumul);

	return AB;
}
/*	TODO
///2D
//PARTITIONS NOT ALLOCATED
spmat* CAT(spmmRowByRow2DBlocksAllocated_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
///SP3MM
spmat* CAT(sp3mmRowByRowPair_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
////////Sp3MM direct
///1D
spmat* CAT(sp3mmRowByRowMerged_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
*/
