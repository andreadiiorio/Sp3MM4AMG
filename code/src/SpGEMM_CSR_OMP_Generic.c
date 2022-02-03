//Developped by     Andrea Di Iorio - 0277550
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO quick hold few CSR cols partition sizes
#include <omp.h>

#include "SpMMMulti.h"
#include "SpMMUtilsMulti.h"
#include "sparseUtilsMulti.h"
#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

#pragma message( "compiling SpGEMM_CSR_OMP_Generic.c with OFF_F as:" STR(OFF_F) )
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif

////inline exports
///multi implmentation functions
void CAT(scSparseVectMul_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen, THREAD_AUX_VECT* aux);
void CAT(scSparseVectMulPart_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen,ulong startIdx,THREAD_AUX_VECT* aux);
void CAT(_scRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux);
void CAT(scSparseRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux);
ulong* CAT(spGEMMSizeUpperbound_,OFF_F)(spmat* A,spmat* B);

///single implmentation functions
SPGEMM_ACC* initSpGEMMAcc(ulong entriesNum, ulong accumulatorsNum);
void freeSpGEMMAcc(SPGEMM_ACC* acc);
void sparsifyDenseVect(SPGEMM_ACC* acc,THREAD_AUX_VECT* accV,SPACC* accSparse, ulong startColAcc);
int mergeRowsPartitions(SPACC* rowsParts,spmat* mat,CONFIG* conf);
int mergeRows(SPACC* rows,spmat* mat);
THREAD_AUX_VECT* _initAccVectors_monoalloc(ulong num,ulong size); //TODO PERF WITH NEXT
int _allocAuxVect(THREAD_AUX_VECT* v,ulong size);
void _resetAccVect(THREAD_AUX_VECT* acc);
void _freeAccVectorsChecks(THREAD_AUX_VECT* vectors,ulong num); 
void freeAccVectors(THREAD_AUX_VECT* vectors,ulong num);

//void C_FortranShiftIdxs(spmat* outMat);
//void Fortran_C_ShiftIdxs(spmat* m);

//global vars	->	audit
double Start,End,Elapsed,ElapsedInternal;

//////////////////////// COMPUTE CORE /////////////////////////////////////////
////////Sp3MM as 2 x SpMM
/////UpperBound 
///1D
spmat* CAT(spgemmRowByRow_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trows of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    ulong* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = CAT(spGEMMSizeUpperbound_,OFF_F) (A,B)))   goto _err;
    
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;

    ((CHUNKS_DISTR_INTERF)	cfg->chunkDistrbFunc) (AB->M,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(acc)
    for (ulong r=0;  r<A->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        acc = accVects + omp_get_thread_num();
        for (ulong c=A->IRP[r]-OFF_F; c<A->IRP[r+1]-OFF_F; c++) //row-by-row formul
            CAT(scSparseRowMul_,OFF_F)(A->AS[c], B, A->JA[c]-OFF_F, acc);
        //trasform accumulated dense vector to a CSR row
        sparsifyDenseVect(outAccumul,acc,outAccumul->accs + r,0);
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
    if(outAccumul)  freeSpGEMMAcc(outAccumul);

    return AB;
}

spmat* CAT(spgemmRowByRow1DBlocks_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trowBlocks of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    ulong* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = CAT(spGEMMSizeUpperbound_,OFF_F)(A,B)))   goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->gridRows,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;
   
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
            sparsifyDenseVect(outAccumul,acc,outAccumul->accs + r,0);
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
    if(outAccumul)  freeSpGEMMAcc(outAccumul);

    return AB;
}

///2D
//PARTITIONS NOT ALLOCATED
spmat* CAT(spgemmRowByRow2DBlocks_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){ 
    DEBUG printf("spgemm\trowBlocks of A ,\tcolBlocks of B\tM=%luxN=%lu\n",A->M,B->N);
    ulong* offsets = NULL;   //B group columns starting offset for each row
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    SPGEMM_ACC* outAccumul=NULL;
    ulong*   rowsSizes=NULL;
    if (!AB)    goto _err;
    if (!(rowsSizes = CAT(spGEMMSizeUpperbound_,OFF_F)(A,B)))   goto _err;
     
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, subRowsN=B->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    //aux vectors  
    ////get offsets for B column groups 
    if (!(offsets = CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(B,cfg->gridCols)))
        goto _err;

    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],subRowsN)))   goto _err;

    ulong bPartLen,bPartID,bPartOffset;//B partition acces aux vars
    
    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (gridSize,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    ulong tileID,t_i,t_j;                            //for aux vars
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
                bPartOffset = offsets[ bPartID ];
                bPartLen    = offsets[ bPartID + 1 ] - bPartOffset;

                CAT(scSparseVectMulPart_,OFF_F)(A->AS[j],B->AS+bPartOffset,
                  B->JA+bPartOffset,bPartLen,startCol,accV);
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,cfg->gridCols);
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   checkOverallocPercent(rowsSizes,AB);
    goto _free;

    _err:
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    if (rowsSizes)   free(rowsSizes);
    if (offsets)     free(offsets);
    if (accVectors)  freeAccVectors(accVectors,gridSize);
    if (outAccumul)  freeSpGEMMAcc(outAccumul);
    
    return AB;
        
}

spmat* CAT(spgemmRowByRow2DBlocksAllocated_,OFF_F)(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trowBlocks of A,\tcolBlocks (allcd) of B\tM=%luxN=%lu\n",A->M,B->N);
    spmat *AB = NULL, *colPartsB = NULL, *colPart;
    ulong*   rowsSizes=NULL;
    //aux vectors  
    SPGEMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    if (!(AB = allocSpMatrix(A->M,B->N)))           goto _err;
    if (!(rowsSizes = CAT(spGEMMSizeUpperbound_,OFF_F)(A,B)))   goto _err;
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, subRowsN=B->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////B cols  partition in CSRs
    if (!(colPartsB = CAT(colsPartitioningUnifRanges_,OFF_F)(B,cfg->gridCols)))  goto _err;
    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],subRowsN)))   goto _err;
    
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
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
	#if OFF_F != 0
	C_FortranShiftIdxs(AB);
	#endif
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   checkOverallocPercent(rowsSizes,AB);
    goto _free;

    _err:
    ERRPRINT("spgemmRowByRow2DBlocksAllocated failed\n");
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    if (colPartsB){
        for (ulong i=0; i<cfg->gridCols; i++)   freeSpmatInternal(colPartsB+i);
        free(colPartsB);
    }
    if (rowsSizes)   free(rowsSizes);
    if (accVectors)  freeAccVectors(accVectors,gridSize);
    if (outAccumul)  freeSpGEMMAcc(outAccumul);
    
    return AB;
        
}
///SP3MM
spmat* CAT(sp3gemmRowByRowPair_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPGEMM_INTERF spgemm){
    
    double end,start,elapsed,partial,flops;
    spmat *RAC = NULL, *out = NULL;
   
    if (!spgemm){
        //TODO runtime on sizes decide witch spgemm implementation to use if not given
        spgemm = &CAT(spgemmRowByRow2DBlocks_,OFF_F);
    }
    /* TODO 
    alloc dense aux vector, reusable over 3 product 
    TODO arrays sovrallocati per poter essere riusati nelle 2 SpGEMM
    ulong auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    */
    
    start = omp_get_wtime();
    /// triple product as a pair of spgemm
    if (!(RAC = spgemm(R,AC,cfg)))      goto _free;
    AUDIT_INTERNAL_TIMES                partial = End - Start;
    if (!(out = spgemm(RAC,P,cfg)))     goto _free;
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
/////Precise
///1D
spmat* CAT(sp3gemmRowByRowMerged_,OFF_F)(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPGEMM_INTERF spgemm){
    ulong* rowsSizes = NULL;
    SPGEMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectorsR_AC=NULL,*accVectorsRAC_P=NULL,*accRAC,*accRACP;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* out = allocSpMatrix(R->M,P->N);
    if (!out)   goto _err;
    /*TODO 3GEMM VERSION COMPUTE OUT ALLOC :  
     -> \forall RAC.row -> hashmap{col=True}->(AVL||RBTHREE); upperBound std col RAC.rows.cols in hashmap || SYM_bis
     * NB: UP per RACP => NN note dimensioni righe precise => stesso approccio riservazione spazio di spgemm ( fetch_and_add )
     *     SYM_BIS ==> note dimensioni righe => 
     *          1) pre riservazione spazio per righe -> cache allignement per threads 
                 -(sc. static & blocco di righe allineato a cache block successivo a blocco righe precedente)
                 -(sc. dynamic& righe tutte allineate a cache block (NO OVERLAPS!) -> huge overhead ?
     *          2) pre riservazione spazio righe diretamente in out CSR
                    -> probabili cache blocks overlap; salvo costo di P.M memcpy
    */
    if (!(rowsSizes = CAT(spGEMMSizeUpperbound_,OFF_F)(R,AC)))   goto _err;	///TODO TOO LOOSE UB...INTEGRATE RBTREE FOR SYM->PRECISE
    ///aux structures alloc 
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[R->M],P->M)))  goto _err; //TODO size estimated with RAC mat
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
        sparsifyDenseVect(outAccumul,accRACP,outAccumul->accs+r,0);
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
    DEBUG                   checkOverallocPercent(rowsSizes,out);
    goto _free;

    _err:
    if(out) freeSpmat(out);
    out = NULL;
    _free:
    if(rowsSizes)       free(rowsSizes);
    if(accVectorsR_AC)  freeAccVectors(accVectorsR_AC,cfg->threadNum);
    if(accVectorsRAC_P) freeAccVectors(accVectorsRAC_P,cfg->threadNum);
    if(outAccumul)      freeSpGEMMAcc(outAccumul);
    
    return out;
}

/////Precise
///1D
