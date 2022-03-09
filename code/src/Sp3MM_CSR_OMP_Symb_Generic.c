//Developped by     Andrea Di Iorio - 0277550
#pragma message( "compiling SpMM_CSR_OMP_Generic.c with OFF_F as:" STR(OFF_F) )
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

#ifndef SP3MM_OMP_SYMB
#define SP3MM_OMP_SYMB
static inline int allocCSRSpMatSymbStep(spmat* m,idx_t* rowSizes){
	//write IRP
	idx_t r,cumulSize;
	for (r=0,cumulSize=0; r<m->M; cumulSize += rowSizes[r++])	m->IRP[r] 	= cumulSize;
		//TODO INNER FOR UPDATETING CUMUL FOR 2D PARTITOONING VERSIONS
	m->IRP[m->M] 	= cumulSize;
	m->NZ			= cumulSize;
	DEBUGCHECKS		assert(cumulSize == rowSizes[m->M]);

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
	if (!(rowsSizes = CAT(SpMM_Symb___,OFF_F) (A,B)))	goto _err;
	if (allocCSRSpMatSymbStep(AB,rowsSizes))			goto _err;
	
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
	if (!(rowsSizes = CAT(SpMM_Symb___,OFF_F) (A,B)))	goto _err;
	if (allocCSRSpMatSymbStep(AB,rowsSizes))			goto _err;
	
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
/*	TODO
///2D
//PARTITIONS NOT ALLOCATED
spmat* CAT(spmmRowByRow2DBlocks_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){ 
spmat* CAT(spmmRowByRow2DBlocksAllocated_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
///SP3MM
spmat* CAT(sp3mmRowByRowPair_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
////////Sp3MM direct
///1D
spmat* CAT(sp3mmRowByRowMerged_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
*/
