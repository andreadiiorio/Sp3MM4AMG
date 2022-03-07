/*
 * CSR Sp[3]MM Symbolic step implementations
 * target: 	compute the output matrix size and the row lens for preallocation
 * 			direct write out partial results
 * See interfaces in respective header
 */


#pragma message( "compiling Sp3MM_CSR_OMP_Symb_Generic.c with config as:" \
	STR(OFF_F) " - " STR(OUT_IDXS) " - " STR(COL_PARTS) )
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

///setup aux macros for different signatures implementation via #if arith expr
#pragma push_macro("OUT_IDXS")
#pragma push_macro("_OUT_IDXS")
#pragma push_macro("COL_PARTS")
#pragma push_macro("_COL_PARTS")

#ifdef OUT_IDXS 
	#define _OUT_IDXS  	TRUE
#else   
	#define _OUT_IDXS 	FALSE
	#define OUT_IDXS	_UNDEF
#endif
#ifdef COL_PARTS
	#define _COL_PARTS	TRUE
#else   
	#define _COL_PARTS	FALSE
	#define COL_PARTS	_UNDEF
#endif
///

//////SpMM - rowByrow
idx_t CAT4(SpMM_Row_Symb_,OUT_IDXS,COL_PARTS,OFF_F)  
  (
   idx_t* aRowJA, idx_t aRowLen, spmat* b,rbRoot* root, rbNode* nodes
   #if _OUT_IDXS  == TRUE && !defined OUT_IDXS_RBTREE_NODES 
   ,idx_t* outIdxs
   #endif
   #if _COL_PARTS == TRUE
   ,ushort gridCols,idx_t* rowColPartsLens
   #endif
  )
{
	//Compute resulting ab's row non zero indexes and total lenght
	idx_t abRowLen = 0;		//mul.result row len,	return value
	for ( idx_t i=0,c_a,inserted; i < aRowLen; i++ ){	//for each entry in a's row
		c_a = aRowJA[i]-OFF_F;
		//gather diffrent nnz indexes in corresponding b's `c_a`th row 
		for ( idx_t j = b->IRP[c_a]-OFF_F,c_b; j < b->IRP[c_a+1]-OFF_F; j++ ){ 
			c_b = b->JA[j]-OFF_F;
			//check if c_b is nonzero index for mul.result row
			inserted 	 =  
			  #ifdef RB_CACHED_INSERT
			  rbInsertCachedNewKey
			  #else
			  rbInsertNewKey
			  #endif
			  (root, nodes+abRowLen, c_b);
			abRowLen	+=	inserted; //inserted needed just after this
			/*	LESS EFFICIENT THEN BELOW (here no memory of last colPart)
			#if _COL_PARTS == TRUE //keep track of which col partition is c_b in
			if (inserted)
			 rowColPartsLens[ matchingUnifRangeIdx(c_b, b->N, gridCols) ]++;
			#endif */
		}
	}
	#if 	_OUT_IDXS == T  && defined OUT_IDXS_RBTREE_NODES
	/* return the non zero indexes of the mul.result row
 	 * sorting inplace the nodes inserted in the rbtree */
	sortRbNode(nodes,abRowLen);
	#elif	_OUT_IDXS == T || _COL_PARTS == T
	uint i=0;
	idx_t k;
	#if _COL_PARTS == T
	//colParts aux vars
	idx_t _colBlock = abRowLen / gridCols, _colBlockRem = abRowLen % gridCols;
	ushort gc=0;
	idx_t gcStartCol = unifRemShareStart(gc,_colBlock,_colBlockRem);
	idx_t gcEndCol = unifRemShareEnd(gc,_colBlock,_colBlockRem);
	#endif	//_COL_PARTS == T
  	for (struct rb_node* n = rb_first(&root->rb_root); n; n = rb_next(n)){
		k = rb_entry(n,rbNode,rb)->key;
		#if _OUT_IDXS == T
		//return the mul.result nnz index inside the rbNodes
		outIdxs[ i++ ] = k;
		#endif
		#if _COL_PARTS == T
		while (k >= gcEndCol ){	//see if the idx is in another col partition
								//	TODO also = since gcEndCol as k is 0based
			gcEndCol = 	unifRemShareEnd(gc ,_colBlock, _colBlockRem);
			gc++;
			DEBUGCHECKS{ assert( gc < gridCols ); }
		}
		rowColPartsLens[gc]++;
		#endif //_COL_PARTS == T
	}
	#endif	//_OUT_IDXS == T ... _COL_PARTS == T
	/*DEBUGCHECKS{	//TODO PRINT NNZ INDEXES FOR MANUAL (PAINFUL CHECK)
		idx_t k;
  		for (struct rb_node* n = rb_first(&root->rb_root); n; n = rb_next(n)){
			k = rb_entry(n,rbNode,rb)->key;
			printf("%lu, ",k);
		}
		printf("\n");
	}*/

	return abRowLen;
}

idx_t* CAT4(SpMM_Symb_,OUT_IDXS,COL_PARTS,OFF_F) 
  (
   spmat* a, spmat* b
   #if _OUT_IDXS  == TRUE
   ,idx_t*** outIdxs
   #endif
   #if _COL_PARTS == TRUE
   ,ushort gridCols, idx_t** rowColPartsLens
   #endif
  )
{

	///initial allocations
	rbRoot* rbRoots = NULL;	rbNode* rbNodes	= NULL;
	idx_t *rowLens=NULL,*upperBoundedRowsLens=NULL,*upperBoundedSymMat=NULL;
	
	if ( !(rowLens = malloc(sizeof(*rowLens) * (a->M+1))) ){ 
		ERRPRINT("SpMM_Symb_ rowLens malloc errd\n");
		goto _err;
	}
	if (!(upperBoundedRowsLens = CAT(spMMSizeUpperbound_,OFF_F)(a,b)))
		goto _err;
	#if _OUT_IDXS  == TRUE
	if (!(*outIdxs = malloc(sizeof(**outIdxs) * a->M))){
		ERRPRINT("SpMM_Symb_ outIdxs malloc errd\n");
		goto _err;
	}
	if (!(upperBoundedSymMat   = malloc(
	  sizeof(*upperBoundedSymMat)*upperBoundedRowsLens[a->M]))){
		ERRPRINT("SpMM_Symb_ upperBoundedSymMat malloc errd\n");
		goto _err;
	}
	//write rows' start pointer from full matrix JA allocated
	for (idx_t i=0,cumul=0; i<a->M; cumul += upperBoundedRowsLens[i++])
		*outIdxs[i] = upperBoundedSymMat + cumul; 
	#endif	//#if _OUT_IDXS  == TRUE
	#if _COL_PARTS == TRUE
	if (!(*rowColPartsLens = malloc(a->M * gridCols * sizeof(**rowColPartsLens)))){
		ERRPRINT("SpMM_Symb_ rowColPartsLens malloc errd\n");
		goto _err;
	}
	#endif //_COL_PARTS
	//rbTrees for index keeping
	idx_t maxRowLen = reductionMaxSeq(upperBoundedRowsLens, a->M);
	uint maxThreads	= omp_get_max_threads();	//TODO FROM CFG
	rbRoots 		= malloc(maxThreads * sizeof(*rbRoots));
	rbNodes			= calloc(maxThreads * maxRowLen, sizeof(*rbNodes));
	if( !rbRoots || !rbNodes ){
		ERRPRINT("SpMM_Symb_ threads' aux rbTree mallocs errd\n");
		goto _err;
	}
	//init roots
	for (uint i=0; i<maxThreads; i++)	rbRoots[i] = RB_ROOT_CACHED;
	///rows parallel compute
	idx_t* aRow;
	idx_t  aRowLen,rLen,abCumulLen=0;
	int tid;
	rbRoot* tRoot;	rbNode* tNodes;
	#pragma omp parallel for schedule(static) \
	  private(aRow,aRowLen,rLen, tRoot,tNodes,tid) reduction(+:abCumulLen)
	for(idx_t r=0; r<a->M; r++){
		aRow	= a->JA + a->IRP[r]-OFF_F;
		aRowLen	= a->IRP[r+1] - a->IRP[r];
		tid 	= omp_get_thread_num();
		tRoot 	= rbRoots + tid;
		tNodes  = rbNodes + tid * maxRowLen;

		rLen = CAT4(SpMM_Row_Symb_,OUT_IDXS,COL_PARTS,OFF_F)  
			(
				aRow,aRowLen,b,tRoot,tNodes
   				#if _OUT_IDXS  == TRUE && !defined OUT_IDXS_RBTREE_NODES 
				,*outIdxs[r]
				#endif
				#if _COL_PARTS == TRUE
				,gridCols, (*rowColPartsLens) + IDX2D(r,0,gridCols)
				#endif
			);
		rowLens[r]  = rLen;
		abCumulLen += rLen;
		//reset rb roots and nodes for next row symb product
		*tRoot = RB_ROOT_CACHED;
		memset(tNodes,0,rLen * sizeof(*tNodes));
	}
	rowLens[a->M]	= abCumulLen;
	return rowLens;

	_err:
	if (rowLens)				free(rowLens);
	if (rbRoots)				free(rbRoots);
	if (rbNodes)				free(rbNodes);
	#if _OUT_IDXS	== TRUE
	if (*outIdxs)				free(*outIdxs);
	if (upperBoundedRowsLens)	free(upperBoundedRowsLens);
	if (upperBoundedSymMat)		free(upperBoundedSymMat);
	#endif
	#if _COL_PARTS	== TRUE
	if (rowColPartsLens)		free(rowColPartsLens);
	#endif
	return NULL;
}


//////Sp3MM - rowByrowByrow
//TODO COL_PARTS VERSION?
#if !defined COL_PARTS && defined OUT_IDXS
idx_t CAT3(Sp3MM_Row_Symb_,OUT_IDXS,OFF_F) (idx_t* aRowJA,idx_t aRowLen,
  spmat* b,spmat* c, rbRoot* root,rbNode* nodes, idx_t* abRowJATmp
  #if _OUT_IDXS  == TRUE && !defined  OUT_IDXS_RBTREE_NODES 
  ,idx_t* outIdxs
  #endif
  )
{
	rb_node* n;
	idx_t abRowLen	= CAT4(SpMM_Row_Symb_,OUT_IDXS_ON,_UNDEF,OFF_F) 
	  (aRowJA,aRowLen,b,root,nodes,abRowJATmp);
	cleanRbNodes(root, nodes, abRowLen);
	idx_t abcRowLen	= CAT4(SpMM_Row_Symb_,OUT_IDXS_ON,_UNDEF,OFF_F)
	  (abRowJATmp,abRowLen,c,root,nodes,abRowJATmp);

	#if 	_OUT_IDXS == TRUE
	#ifndef OUT_IDXS_RBTREE_NODES	
	//return the mul.result nnz index inside the rbNodes
	uint i=0;
	rbNodeOrderedVisit(n,root)	outIdxs[ i++ ] = rb_entry(n,rbNode,rb)->key;
	#else
	/* return the non zero indexes of the mul.result row
 	 * sorting inplace the nodes inserted in the rbtree */
	sortRbNode(nodes,abcRowLen);
	#endif
	#endif
	return abcRowLen;
}

idx_t* CAT3(Sp3MM_Symb_,OUT_IDXS,OFF_F) (spmat* a,spmat* b,spmat* c
  #if _OUT_IDXS  == TRUE
  ,idx_t*** outIdxs
  #endif
  )
{
	///initial allocations
	idx_t* rowLens = malloc(sizeof(*rowLens) * (a->M +1) ); //to return
	if (!rowLens){
		ERRPRINT("SpMM_Symb_ rowLens malloc errd\n");
		goto _err;
	}
	
	#if _OUT_IDXS  == TRUE
	idx_t *abUpperBoundedRowsLens = NULL, *upperBoundedSymMat = NULL;
	if (!(*outIdxs = malloc(sizeof(**outIdxs) * a->M))){
		ERRPRINT("SpMM_Symb_ outIdxs malloc errd\n");
		goto _err;
	}
	if (!(abUpperBoundedRowsLens = CAT(spMMSizeUpperbound_,OFF_F)(a,b)))	
		goto _err;
  	/*TODO TODO instead of doing one sym product first to have a correct UB
     *	use an heuristics here to get output matrix size
     */
	idx_t abcUBSize = abUpperBoundedRowsLens[a->M] * SP3MM_UB_HEURISTIC;
	if (!(upperBoundedSymMat=malloc(sizeof(*upperBoundedSymMat)*abcUBSize))){
		ERRPRINT("SpMM_Symb_ upperBoundedSymMat malloc errd\n");
		goto _err;
	}
	//TODO heuristic TO UB rows bounds
	for (idx_t i=0,cumul=0; i<a->M; 
	  cumul += SP3MM_UB_HEURISTIC * upperBoundedRowsLens[i++])
		*outIdxs[i] = upperBoundedSymMat + cumul; 
  	#endif	//#if _OUT_IDXS  == TRUE
	//rbTrees for index keeping
	uint maxThreads 		= omp_get_max_threads(); //TODO FROM CFG
	idx_t abMaxRowLen 	= reductionMaxSeq(abUpperBoundedRowsLens, a->M);
	#ifdef 	  HEURISTICS_UB
	idx_t maxRowLenUB 	= abMaxRowLen * SP3MM_UB_HEURISTIC; //TODO UB HEURISTC
	#else
	idx_t maxRowLenUB 	= c->N;
	#endif	//HEURISTICS_UB
	//threads local bufs
	rbRoot* rbRoots 	= malloc(maxThreads * sizeof(*rbRoots));
	rbNode* rbNodes		= malloc(maxThreads * maxRowLenUB * sizeof(*rbNodes));
	rbNode* abRowsJATmp	= malloc(maxThreads*maxRowLenUB*sizeof(*abRowsJATmp));
	if( !rbRoots || !rbNodes ){
		ERRPRINT("SpMM_Symb_ threads' aux rbTree mallocs errd\n");
		goto _err;
	}
	///rows parallel compute
	idx_t* aRow;
	idx_t  aRowLen,rLen,outCumulLen;
	//threads local pointers
	int tid;	rbRoot* tRoot;	rbNode* tNodes;idx_t* tABRowJATmp;
	#pragma omp parallel for schedule(static) \
	private(aRow,aRowLen,rLen, tRoot,tNodes,tid) reduction(+:abCumulLen)
	for(idx_t r=0; r<a->M; r++){
		aRow		= a->JA + a->IRP[r]-OFF_F;
		aRowLen		= a->IRP[r+1] - a->IRP[r];
		tid 		= omp_get_thread_num();
		tRoot 		= rbRoots + tid;
		tNodes  	= rbNodes + tid * maxRowLenUB;
		tABRowJATmp	= abRowsJATmp + tid * maxRowLenUB;
		rLen = CAT4(Sp3MM_Row_Symb_,OUT_IDXS,OFF_F)
		  (aRow,aRowLen,b,c,tRoot,tNodes,tABRowJATmp,
		  	#if _OUT_IDXS == TRUE
			*outIdxs[r]
			#endif
		  );
		
	}

	return rowLens;
}

#endif	//#if !defined COL_PARTS && defined OUT_IDXS


///restore aux macros entry state
//#undef _OUT_ID
//#undef _COL_PARTS
#pragma pop_macro("OUT_IDXS")
#pragma pop_macro("_OUT_IDXS")
#pragma pop_macro("COL_PARTS")
#pragma pop_macro("_COL_PARTS")




