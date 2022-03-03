//Developped by     Andrea Di Iorio - 0277550
#pragma message( "compiling SpMM_CSR_OMP_Generic.c with OFF_F as:" STR(OFF_F) )
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif



//////////////////// COMPUTE CORE Sp[3]MM Upperbound //////////////////////////
////////Sp3MM as 2 x SpMM
///1D
spmat* CAT(spmmRowByRow_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spmm\trows of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPMM_ACC* outAccumul=NULL;
    idx_t* rowsSizes = NULL;
    ///init AB matrix with SPMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = CAT(spMMSizeUpperbound_,OFF_F) (A,B)))   goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;
    #if SPARSIFY_PRE_PARTITIONING == T
	//prepare sparse accumulators with U.Bounded rows[parts] starts
	SPACC* accSp;
	for( idx_t r=0,rSizeCumul=0; r<AB->M; rSizeCumul += rowsSizes[r++]){
		accSp 		= outAccumul->accs+r;
		accSp->JA 	= outAccumul->JA + rSizeCumul;
		accSp->AS 	= outAccumul->AS + rSizeCumul;
		//accSp->len	= rowsSizes[r];
	}
	#endif

    ((CHUNKS_DISTR_INTERF)	cfg->chunkDistrbFunc) (AB->M,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(acc)
    for (ulong r=0;  r<A->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        acc = accVects + omp_get_thread_num();
        for (ulong c=A->IRP[r]-OFF_F; c<A->IRP[r+1]-OFF_F; c++) //row-by-row formul
            CAT(scSparseRowMul_,OFF_F)(A->AS[c], B, A->JA[c]-OFF_F, acc);
        //trasform accumulated dense vector to a CSR row
        #if SPARSIFY_PRE_PARTITIONING == T
		_sparsifyUB(acc,outAccumul->accs+r,0);
		#else
        sparsifyUBNoPartsBounds(outAccumul,acc,outAccumul->accs + r,0);
		#endif
        _resetAccVect(acc);   //rezero for the next A row
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,AB))    goto _err;
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
    if(accVects)    freeAccVectors(accVects,cfg->threadNum);
    if(outAccumul)  freeSpMMAcc(outAccumul);

    return AB;
}

spmat* CAT(spmmRowByRow1DBlocks_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spmm\trowBlocks of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPMM_ACC* outAccumul=NULL;
    idx_t* rowsSizes = NULL;
    ///init AB matrix with SPMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = CAT(spMMSizeUpperbound_,OFF_F)(A,B)))   goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->gridRows,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;
    #if SPARSIFY_PRE_PARTITIONING == T
	//prepare sparse accumulators with U.Bounded rows[parts] starts
	SPACC* accSp;
	for( idx_t r=0,rSizeCumul=0; r<AB->M; rSizeCumul += rowsSizes[r++]){
		accSp = outAccumul->accs+r;
		accSp->JA = outAccumul->JA + rSizeCumul;
		accSp->AS = outAccumul->AS + rSizeCumul;
	}
	#endif
   
    //perform Gustavson over rows blocks -> M / @cfg->gridRows
    ulong rowBlock = AB->M/cfg->gridRows, rowBlockRem = AB->M%cfg->gridRows;
    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (cfg->gridRows,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    ulong b,startRow,block; //omp for aux vars
    #pragma omp parallel for schedule(runtime) private(acc,startRow,block)
    for (b=0;   b < cfg->gridRows; b++){
        block      = UNIF_REMINDER_DISTRI(b,rowBlock,rowBlockRem);
        startRow   = UNIF_REMINDER_DISTRI_STARTIDX(b,rowBlock,rowBlockRem);
       
        DEBUGPRINT{
            fflush(NULL);
            printf("block %lu\t%lu:%lu(%lu)\n",b,startRow,startRow+block-1,block);
            fflush(NULL);
        }
        //row-by-row formulation in the given row block
        for (ulong r=startRow;  r<startRow+block;  r++){
            //iterate over nz entry index c inside current row r
            acc = accVects + b;
            for (ulong c=A->IRP[r]-OFF_F; c<A->IRP[r+1]-OFF_F; c++) 
                CAT(scSparseRowMul_,OFF_F)(A->AS[c], B, A->JA[c]-OFF_F, acc);
            //trasform accumulated dense vector to a CSR row
        	#if SPARSIFY_PRE_PARTITIONING == T
			_sparsifyUB(acc,outAccumul->accs+r,0);
			#else
        	sparsifyUBNoPartsBounds(outAccumul,acc,outAccumul->accs + r,0);
			#endif
            _resetAccVect(acc);   //rezero for the next A row
        }
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,AB))    goto _err;
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
    if(accVects)    freeAccVectors(accVects,cfg->gridRows);
    if(outAccumul)  freeSpMMAcc(outAccumul);

    return AB;
}

///2D
//PARTITIONS NOT ALLOCATED
spmat* CAT(spmmRowByRow2DBlocks_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){ 
    DEBUG printf("spmm\trowBlocks of A ,\tcolBlocks of B\tM=%luxN=%lu\n",A->M,B->N);
    idx_t* bColOffsets = NULL;   //B group columns starting offset for each row
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    SPMM_ACC* outAccumul=NULL;
    idx_t*    rowsPartsSizes=NULL;
    if (!AB)    goto _err;
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, aSubRowsN=A->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////get bColOffsets for B column groups 
    if (!(bColOffsets = CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(B,cfg->gridCols)))
		goto _err;
	#if SPARSIFY_PRE_PARTITIONING == T
	uint rowsPartsSizesN = aSubRowsN;
    if (!(rowsPartsSizes = CAT(spMMSizeUpperboundColParts_,OFF_F)(A,B,cfg->gridCols,bColOffsets)))
	#else
	uint rowsPartsSizesN = AB->M;
    if (!(rowsPartsSizes = CAT(spMMSizeUpperbound_,OFF_F)(A,B)))
	#endif
		goto _err;
     
    //aux vectors  

    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpMMAcc(rowsPartsSizes[rowsPartsSizesN],aSubRowsN)))   
		goto _err;
	#if SPARSIFY_PRE_PARTITIONING == T
	//prepare sparse accumulators with U.Bounded rows[parts] starts
	SPACC* accSp;
	for( idx_t i=0,rSizeCumul=0; i<aSubRowsN; rSizeCumul += rowsPartsSizes[i++]){
		accSp 		= outAccumul->accs+i;
		accSp->JA 	= outAccumul->JA + rSizeCumul;
		accSp->AS 	= outAccumul->AS + rSizeCumul;
	}
	//memset(outAccumul->AS,0,sizeof(double)*rowsSizes[AB->M]);memset(outAccumul->JA,0,sizeof(idx_t)*rowsSizes[AB->M]);
	#endif
    
    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (gridSize,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    ulong tileID,t_i,t_j;                            //for aux vars
    ulong bPartLen,bPartID,bPartOffset;//B partition acces aux vars
    #pragma omp parallel for schedule(runtime) \
      private(accV,accRowPart,rowBlock,colBlock,startRow,startCol,\
      bPartLen,bPartID,bPartOffset,t_i,t_j)
    for (tileID = 0; tileID < gridSize; tileID++){
        ///get iteration's indexing variables
        //tile index in the 2D grid of AB computation TODO OMP HOW TO PARALLELIZE 2 FOR
        t_i = tileID/cfg->gridCols;  //i-th row block
        t_j = tileID%cfg->gridCols;  //j-th col block
        //get tile row-cols group FAIR sizes
        rowBlock = UNIF_REMINDER_DISTRI(t_i,_rowBlock,_rowBlockRem); 
        startRow = UNIF_REMINDER_DISTRI_STARTIDX(t_i,_rowBlock,_rowBlockRem);
        startCol = UNIF_REMINDER_DISTRI_STARTIDX(t_j,_colBlock,_colBlockRem);
        
        accV = accVectors + tileID; 
         
        DEBUGPRINT{
            fflush(NULL);
            colBlock = UNIF_REMINDER_DISTRI(t_j,_colBlock,_colBlockRem);
            printf("rowBlock [%lu\t%lu:%lu(%lu)]\t",t_i,startRow,startRow+rowBlock-1,rowBlock);
            printf("colBlock [%lu\t%lu:%lu(%lu)]\n",t_j,startCol,startCol+colBlock-1,colBlock);
            fflush(NULL);
        }
        ///AB[t_i][t_j] block compute
        for (ulong r=startRow;  r<startRow+rowBlock;  r++){
            //iterate over nz col index j inside current row r
            //row-by-row restricted to colsubset of B to get AB[r][:colBlock:]
            for (ulong j=A->IRP[r]-OFF_F,c; j<A->IRP[r+1]-OFF_F; j++){
                //get start of B[A->JA[j]][:colBlock:]
                c = A->JA[j]-OFF_F; // col of nnz in A[r][:] <-> target B row
                bPartID     = IDX2D(c,t_j,cfg->gridCols); 
                bPartOffset = bColOffsets[ bPartID ];
                bPartLen    = bColOffsets[ bPartID + 1 ] - bPartOffset;

                CAT(scSparseVectMulPart_,OFF_F)(A->AS[j],B->AS+bPartOffset,
                  B->JA+bPartOffset,bPartLen,startCol,accV);
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,cfg->gridCols);
			#if SPARSIFY_PRE_PARTITIONING == T
			_sparsifyUB(accV,accRowPart,startCol);
			#else
            sparsifyUBNoPartsBounds(outAccumul,accV,accRowPart,startCol);
			#endif
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   
	  checkOverallocRowPartsPercent(rowsPartsSizes,AB,cfg->gridCols,bColOffsets);
    goto _free;

    _err:
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    free(rowsPartsSizes);
    free(bColOffsets);
    if (accVectors)  freeAccVectors(accVectors,gridSize);
    if (outAccumul)  freeSpMMAcc(outAccumul);
    
    return AB;
        
}

spmat* CAT(spmmRowByRow2DBlocksAllocated_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spmm\trowBlocks of A,\tcolBlocks (allcd) of B\tM=%luxN=%lu\n",A->M,B->N);
    spmat *AB = NULL, *colPartsB = NULL, *colPart;
    idx_t*   rowsPartsSizes=NULL;
    //aux vectors  
    SPMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    if (!(AB = allocSpMatrix(A->M,B->N)))           goto _err;

    //2D indexing aux vars
    idx_t gridSize=cfg->gridRows*cfg->gridCols, aSubRowsN=A->M*cfg->gridCols;
	idx_t* bColOffsets = NULL;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////B cols  partition in CSRs
    //if (!(colPartsB = CAT(colsPartitioningUnifRanges_,OFF_F)(B,cfg->gridCols)))  goto _err;
    if (!(colPartsB = CAT(colsPartitioningUnifRangesOffsetsAux_,OFF_F)(B,cfg->gridCols,&bColOffsets)))  goto _err;
	#if SPARSIFY_PRE_PARTITIONING == T
	uint rowsPartsSizesN = aSubRowsN;
    if (!(rowsPartsSizes = CAT(spMMSizeUpperboundColParts_,OFF_F)
	  (A,B,cfg->gridCols,bColOffsets)))   
	#else
	uint rowsPartsSizesN = AB->M;
    if (!(rowsPartsSizes = CAT(spMMSizeUpperbound_,OFF_F)(A,B)))
	#endif
		goto _err;
    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpMMAcc(rowsPartsSizes[rowsPartsSizesN],aSubRowsN)))
		goto _err;
	#if SPARSIFY_PRE_PARTITIONING == T
	//prepare sparse accumulators with U.Bounded rows[parts] starts
	SPACC* accSp;
	for( idx_t i=0,rLenCumul=0; i<aSubRowsN; rLenCumul += rowsPartsSizes[i++]){
		accSp 		= outAccumul->accs+i;
		accSp->JA 	= outAccumul->JA + rLenCumul;
		accSp->AS 	= outAccumul->AS + rLenCumul;
	}
	#endif
    
    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (gridSize,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    ulong tileID,t_i,t_j;    //for aux vars
    #pragma omp parallel for schedule(runtime) \
      private(accV,accRowPart,colPart,rowBlock,colBlock,startRow,startCol,t_i,t_j)
    for (tileID = 0; tileID < gridSize; tileID++){
        ///get iteration's indexing variables
        //tile index in the 2D grid of AB computation TODO OMP HOW TO PARALLELIZE 2 FOR
        t_i = tileID/cfg->gridCols;  //i-th row block
        t_j = tileID%cfg->gridCols;  //j-th col block
        //get tile row-cols group FAIR sizes
        rowBlock = UNIF_REMINDER_DISTRI(t_i,_rowBlock,_rowBlockRem); 
        colBlock = UNIF_REMINDER_DISTRI(t_j,_colBlock,_colBlockRem);
        startRow = UNIF_REMINDER_DISTRI_STARTIDX(t_i,_rowBlock,_rowBlockRem);
        startCol = UNIF_REMINDER_DISTRI_STARTIDX(t_j,_colBlock,_colBlockRem);
        
        colPart = colPartsB + t_j;
        accV = accVectors + tileID; 
         
        DEBUGPRINT{
            fflush(NULL);
            printf("rowBlock [%lu\t%lu:%lu(%lu)]\t",t_i,startRow,startRow+rowBlock-1,rowBlock);
            printf("colBlock [%lu\t%lu:%lu(%lu)]\n",t_j,startCol,startCol+colBlock-1,colBlock);
            fflush(NULL);
        }
        ///AB[t_i][t_j] block compute
        for (ulong r=startRow;  r<startRow+rowBlock;  r++){
            //iterate over nz col index j inside current row r
            //row-by-row restricted to colsubset of B to get AB[r][:colBlock:]
            for (ulong j=A->IRP[r]-OFF_F,c,bRowStart,bRowLen; j<A->IRP[r+1]-OFF_F; j++){
                //get start of B[A->JA[j]][:colBlock:]
                c = A->JA[j]-OFF_F; // column of nnz entry in A[r][:] <-> target B row
                //CAT(scSparseRowMul_,OFF_F)(A->AS[j],colPart,c,accV);//TODO GENERIC VERSION USEFUL
                bRowStart = colPart->IRP[c];
				#ifdef ROWLENS
                bRowLen   = colPart->RL[c];
				#else
                bRowLen   = colPart->IRP[c+1] - bRowStart;
				#endif
                CAT(scSparseVectMulPart_,OFF_F)(A->AS[j],colPart->AS+bRowStart,colPart->JA+bRowStart,
                    bRowLen,startCol,accV);
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,cfg->gridCols);
			#if SPARSIFY_PRE_PARTITIONING == T
			_sparsifyUB(accV,accRowPart,startCol);
			#else
            sparsifyUBNoPartsBounds(outAccumul,accV,accRowPart,startCol);
			#endif
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   
	  checkOverallocRowPartsPercent(rowsPartsSizes,AB,cfg->gridCols,bColOffsets);
    goto _free;

    _err:
    ERRPRINT("spmmRowByRow2DBlocksAllocated failed\n");
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    if (colPartsB){
        for (ulong i=0; i<cfg->gridCols; i++)   freeSpmatInternal(colPartsB+i);
        free(colPartsB);
    }
    free(rowsPartsSizes);
    free(bColOffsets);
    if (accVectors)  freeAccVectors(accVectors,gridSize);
    if (outAccumul)  freeSpMMAcc(outAccumul);
    
    return AB;
        
}
///SP3MM
spmat* CAT(sp3mmRowByRowPair_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
    
    double end,start,elapsed,partial,flops;
    spmat *RAC = NULL, *out = NULL;
   
    if (!spmm){
        //TODO runtime on sizes decide witch spmm implementation to use if not given
        spmm = &CAT(spmmRowByRow2DBlocks_,OFF_F);
    }
    /* TODO 
    alloc dense aux vector, reusable over 3 product 
    TODO arrays sovrallocati per poter essere riusati nelle 2 SpMM
    ulong auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    */
    
    start = omp_get_wtime();
    /// triple product as a pair of spmm
    if (!(RAC = spmm(R,AC,cfg)))	goto _free;
    AUDIT_INTERNAL_TIMES			partial = End - Start;
    if (!(out = spmm(RAC,P,cfg)))	goto _free;
    //
    end = omp_get_wtime();
    ElapsedInternal = End - Start + partial;
    VERBOSE {
        elapsed         = end - start;
        flops = ( 2 * R->NZ * P->NZ * AC->NZ ) / ( elapsed );
        printf("elapsed %le - flops %le",elapsed,flops);
        AUDIT_INTERNAL_TIMES    printf("\tinternalTime: %le",ElapsedInternal);
        printf("\n");
    }
    _free:
    if (RAC)    freeSpmat(RAC);

    return out;
}

////////Sp3MM direct
///1D
spmat* CAT(sp3mmRowByRowMerged_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPMM_INTERF spmm){
    ulong* rowSizes = NULL;
    SPMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectorsR_AC=NULL,*accVectorsRAC_P=NULL,*accRAC,*accRACP;
    ///init AB matrix with SPMM heuristic preallocation
    spmat* out = allocSpMatrix(R->M,P->N);
    if (!out)   goto _err;
    /*TODO 3MM VERSION COMPUTE OUT ALLOC :  
     -> \forall RAC.row -> hashmap{col=True}->(AVL||RBTHREE); upperBound std col RAC.rows.cols in hashmap || SYM_bis
     * NB: UP per RACP => NN note dimensioni righe precise => stesso approccio riservazione spazio di spmm ( fetch_and_add )
     *     SYM_BIS ==> note dimensioni righe => 
     *          1) pre riservazione spazio per righe -> cache allignement per threads 
                 -(sc. static & blocco di righe allineato a cache block successivo a blocco righe precedente)
                 -(sc. dynamic& righe tutte allineate a cache block (NO OVERLAPS!) -> huge overhead ?
     *          2) pre riservazione spazio righe diretamente in out CSR
                    -> probabili cache blocks overlap; salvo costo di P.M memcpy
    */
    if (!(rowSizes = CAT(spMMSizeUpperbound_,OFF_F)(R,AC)))   goto _err;	///TODO TOO LOOSE UB...INTEGRATE RBTREE FOR SYM->PRECISE
    ///aux structures alloc 
    if (!(outAccumul = initSpMMAcc(rowSizes[R->M],P->M)))  goto _err; //TODO size estimated with RAC mat
    if (!(accVectorsR_AC = _initAccVectors(cfg->threadNum,AC->N))){ //TODO LESS || REUSE
        ERRPRINT("accVectorsR_AC init failed\n");
        goto _err;
    }
    if (!(accVectorsRAC_P = _initAccVectors(cfg->threadNum,R->N))){ //TODO LESS || REUSE
        ERRPRINT("accVectorsRAC_P init failed\n");
        goto _err;
    }

    ulong c;
    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (R->M,R,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(accRAC,accRACP,c)
    for (ulong r=0;  r<R->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        accRAC  = accVectorsR_AC  + omp_get_thread_num();
        accRACP = accVectorsRAC_P + omp_get_thread_num();
		//computing (tmp) R*AC r-th row
        for (ulong j=R->IRP[r]-OFF_F; j<R->IRP[r+1]-OFF_F; j++)
            CAT(scSparseRowMul_,OFF_F)(R->AS[j], AC, R->JA[j]-OFF_F, accRAC);
        //forward the computed row
        for (ulong j=0; j<accRAC->nnzIdxLast; j++){
            c = accRAC->nnzIdx[j];    
            CAT(scSparseRowMul_,OFF_F)(accRAC->v[c],P,c,accRACP);
        }
        //trasform accumulated dense vector to a CSR row TODO in UB buff
        sparsifyUBNoPartsBounds(outAccumul,accRACP,outAccumul->accs+r,0);
        _resetAccVect(accRAC);
        _resetAccVect(accRACP);
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,out))    goto _err;
	#if OFF_F != 0
	C_FortranShiftIdxs(out);
	#endif
    AUDIT_INTERNAL_TIMES{
        End=omp_get_wtime();
        ElapsedInternal = End-Start;
    }
    DEBUG                   checkOverallocPercent(rowSizes,out);
    goto _free;

    _err:
    if(out) freeSpmat(out);
    out = NULL;
    _free:
    if(rowSizes)       free(rowSizes);
    if(accVectorsR_AC)  freeAccVectors(accVectorsR_AC,cfg->threadNum);
    if(accVectorsRAC_P) freeAccVectors(accVectorsRAC_P,cfg->threadNum);
    if(outAccumul)      freeSpMMAcc(outAccumul);
    
    return out;
}
