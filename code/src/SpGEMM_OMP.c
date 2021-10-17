//Developped by     Andrea Di Iorio - 0277550
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif

//inline export here 
spmat* allocSpMatrix(uint rows, uint cols);

//TODO che guadagno si ha ad utilizzare solo la versione generica delle successive 2 funzioni
/*
 * Sparse vector part <->scalar multiplication in a dense output
 * sparse vector part will hold nnz values in @vectVals 
 * with corresponding indexes in @vectIdxs in range [0,@vectLen]
 * resulting vector accumulated in a dense array in @aux->v, along with nnzIdx
 * both of accumulator's dense array and nnzIdx in @aux and has to be big @vectLen
 */
static inline void scSparseVectMul(double scalar,
  double* vectVals,uint* vectIdxs,uint vectLen, THREAD_AUX_VECT* aux){
    for (uint i=0,j; i<vectLen; i++){
        j = vectIdxs[i];
        DEBUGCHECKS{
            if (j>aux->vLen){
                fprintf(stderr,"index %u outside vLen %u\n",j,aux->vLen);
                exit(1);
            }
        }
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))    aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += vectVals[i] * scalar;  //accumulate
    }
}


/*
 * Sparse vector part <->scalar multiplication in a dense output
 * sparse vector part will hold nnz values in @vectVals 
 * with [at least] @vectLen corresponding target idxs 
 * in @vectIdxs starting from @startIdx 
 * Resulting vector accumulated in a dense array in @aux->v, along with nnzIdx
 * all nnz values indexes will be shifted back of @startIdx in @aux
 * both of accumulator's dense array and nnzIdx in @aux and has to be big @vectLen
 */
static inline void scSparseVectMulPart(double scalar,double* vectVals,
  uint* vectIdxs,uint vectLen,uint startIdx,THREAD_AUX_VECT* aux){
    for (uint i=0,j; i<vectLen; i++){
        j = vectIdxs[i] - startIdx; //shift back nnz value index for the accumul
        DEBUGCHECKS{ //TODO REMOVE
            if (j>aux->vLen){
                fprintf(stderr,"index %u outside vLen %u\n",j,aux->vLen);
                exit(1);
            }
        }
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))   aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += vectVals[i] * scalar;  //accumulate
    }
}

///TODO OLD DIRECT USE OF SCALAR<->ROW MULTIPLICATION -- REMOVABLE
static inline void _scRowMul(double scalar,spmat* mat,uint trgtR, THREAD_AUX_VECT* aux){
    for (uint c=mat->IRP[trgtR],j;  c<mat->IRP[trgtR+1];  c++){//scanning trgtRow
        j = mat->JA[c];
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))    aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += mat->AS[c] * scalar;  //accumulate
    }
}
///TODO IMPLEMENT SCALAR<->ROW MUL AS GENERIC SPARSE VECTOR<->SCALAR MUL
static inline void scRowMul(double scalar,spmat* mat,uint trgtR, THREAD_AUX_VECT* aux){
    uint  rowStartIdx = mat->IRP[trgtR],rowLen;
#ifdef ROWLENS
    rowLen = mat->RL[trgtR];
#else
    rowLen = mat->IRP[trgtR+1]-rowStartIdx;
#endif
    scSparseVectMul(scalar,mat->AS+rowStartIdx,mat->JA+rowStartIdx,rowLen,aux);
    //scSparseVectMulPart(scalar,mat->AS+rowStartIdx,mat->JA+rowStartIdx,rowLen,0,aux);
    //TODO TODO check impact of generic version use
}


///AUX
/*
 * alloc threads' aux arrays once and split them in threads' structures
 *so free them once from the first thread struct, with the original pointers returned from the alloc
 */
static THREAD_AUX_VECT* _initAccVectors_monoalloc(uint num,uint size){
    THREAD_AUX_VECT* out    = NULL;
    double* vAll            = NULL;
    uint* vAllNzIdx         = NULL;
    if (!(out = calloc(num,sizeof(*out)))){
        ERRPRINT("_initAccVectors aux struct alloc failed\n");
        return NULL;
    }
    if (!(vAll = calloc(num*size,sizeof(*vAll)))) {
        ERRPRINT("_initAccVectors aux dense vectors alloc failed\n");
        goto err;
    }
    if (!(vAllNzIdx = calloc(num*size,sizeof(*vAllNzIdx)))) {
        ERRPRINT("_initAccVectors aux dense vectors' idx alloc failed\n");
        goto err;
    }
    for (uint i=0; i<num; i++){
        out[i].vLen        = size; //TODO USELESS INFO?
        out[i].v           = vAll      + i*size;  
        out[i].nnzIdx      = vAllNzIdx + i*size;;
        out[i].nnzIdxLast  = 0;
    }
    return out;
    
    err:
    free(out);
    if (vAll)        free(vAll);
    if (vAllNzIdx)   free(vAllNzIdx);
    return NULL;
}
static inline int _allocAuxVect(THREAD_AUX_VECT* v,uint size){
        v->vLen = size; 
        if (!(v->v = calloc(size,sizeof(*(v->v))))) {
            ERRPRINT("_initAccVectors aux dense vector alloc failed\n");
            return EXIT_FAILURE;
        }
        if (!(v->nnzIdx = calloc(size,sizeof(*(v->nnzIdx))))) {
            ERRPRINT("_initAccVectors aux dense vector' idx alloc failed\n");
            return EXIT_FAILURE;
        }
        v->nnzIdxLast  = 0;
        return EXIT_SUCCESS;
}
//alloc threads' rows accumulators vectors
static THREAD_AUX_VECT* _initAccVectors(uint num,uint size){
    THREAD_AUX_VECT* out    = NULL;
    if (!(out = calloc(num,sizeof(*out)))){
        ERRPRINT("_initAccVectors aux struct alloc failed\n");
        return NULL;
    }
    for (uint i=0; i<num; i++){
        if (_allocAuxVect(out+i,size))   goto _err;
    }
    return out;
    
    _err:
    for (uint i=0; i<num; i++){
        if (out[i].v)       free(out[i].v);
        if (out[i].nnzIdx)  free(out[i].nnzIdx);
    }
    free(out);
    return NULL;
}
static inline void _resetAccVect(THREAD_AUX_VECT* acc){
    memset(acc->v,0,acc->vLen * sizeof(*(acc->v)));
    memset(acc->nnzIdx,0,acc->vLen * sizeof(*(acc->nnzIdx)));
    acc->nnzIdxLast = 0;
}
static void _freeAccVectorsChecks(THREAD_AUX_VECT* vectors,uint num){
    if (!vectors)   return;
    for (uint i=0; i<num; i++){
        if(vectors[i].v)        free(vectors[i].v);
        if(vectors[i].nnzIdx)   free(vectors[i].nnzIdx);
    }
    free(vectors);
}
static void _freeAccVectors(THREAD_AUX_VECT* vectors,uint num){
    for (uint i=0; i<num; i++){
        free(vectors[i].v);
        free(vectors[i].nnzIdx);
    }
    free(vectors);
}


/*
 * sparsify accumulated vector @accV into sparse matrix row [partition] @accRow
 * @accV has non zero values and indexed shifted back of @startColAcc columns
 * TODO renames: accRow & type -> accSparse; rowNNZ -> nnz ... 
 */
static inline int sparsifyDenseVect(THREAD_AUX_VECT* accV,SPMATROW* accRow,
  uint startColAcc){
    uint rowNNZ = accV -> nnzIdxLast;
    accRow -> len = rowNNZ;
    //alloc row nnz element space
    if (!(accRow->AS = malloc(rowNNZ * sizeof(*(accRow->AS))))){
        ERRPRINT("sparsifyDenseVect: accRow->AS malloc error\n");
        goto _err;
    }
    if (!(accRow->JA = malloc(rowNNZ * sizeof(*(accRow->JA))))){
        ERRPRINT("sparsifyDenseVect: accRow->JA malloc error\n");
        goto _err;
    }

    sortuint(accV->nnzIdx,rowNNZ); //sort nnz idx for ordered write

    ///SPARSIFY DENSE ACC.V INTO ROW
    for (uint i=0,j;    i<rowNNZ;   i++){ 
        j = accV -> nnzIdx[i];        //shifted idx of a nnz of sp.Vect accumulator
        accRow -> JA[i] = j + startColAcc;
        accRow -> AS[i] = accV->v[j];
    }
    CONSISTENCY_CHECKS{ //TODO check if nnzIdxs sparsify is good as a full scan
        double* tmpNNZ = calloc(accV->vLen, sizeof(*tmpNNZ)); //TODO OVERFLOW POSSIBLE
        if (!tmpNNZ){ERRPRINT("sparsify check tmpNNZ malloc err\n");goto _err;}
        uint ii=0;
        for (uint i=0; i< accV->vLen; i++){
            if ( accV->v[i] )   tmpNNZ[ii++] = accV->v[i];
        }
        //if (memcmp(tmpNNZ,accRow->AS,rowNNZ*sizeof(*tmpNNZ)))
        if (doubleVectorsDiff(tmpNNZ,accRow->AS,rowNNZ))
            {ERRPRINT("quick sparsify check err\n");free(tmpNNZ);goto _err;}
        if (ii != rowNNZ){
            fprintf(stderr,"quick sparsify nnz num wrong!:%u<->%u\n",ii,rowNNZ);
            free(tmpNNZ);goto _err;
        }
        free(tmpNNZ);
    }
    return EXIT_SUCCESS;
    _err:
    //if(accRow->AS)  free(accRow->AS); //TODO free anyway later
    //if(accRow->JA)  free(accRow->JA);
    return EXIT_FAILURE;    
}

/*
 * merge @conf->gridCols*@mat->M sparse rows partitions into @mat
 * EXPECTED rowsParts @rowsParts to be sorted in accord to the 
 * 2D rowMajor computing grid given in @conf
 * allocd arrays to hold non zero values and indexes into @mat
 */
static inline int mergeRowsPartitions(SPMATROW* rowsParts,spmat* mat,
  CONFIG* conf){
    uint nzNum=0,j,rLen, partNum = mat->M * conf->gridCols;
    ///count nnz entries and alloc arrays for them
    for (uint r=0; r<mat->M; r++){
        for (j=0,rLen=0; j<conf->gridCols; j++)  //for each partition -> get len
            rLen += rowsParts[ IDX2D(r,j,conf->gridCols) ].len;
        nzNum += rLen;
        mat->IRP[r+1] = nzNum;
#ifdef ROWLENS
        mat->RL[r] = rLen;
#endif
    }
    mat->NZ = nzNum;
    if (!(mat->AS = malloc(nzNum * sizeof(*(mat->AS))))){
        ERRPRINT("merged sparse matrix AS alloc errd\n");
        return EXIT_FAILURE;
    }  
    if (!(mat->JA = malloc(nzNum * sizeof(*(mat->JA))))){
        ERRPRINT("merged sparse matrix JA alloc errd\n");
        return EXIT_FAILURE;
    }
    ///POPOLATE WITH ROWS NNZ VALUES AND INDEXES
    for (uint i=0,startOff=0,pLen;  i<partNum;  startOff+=rowsParts[i++].len){
        pLen = rowsParts[i].len;
        memcpy(mat->AS + startOff,rowsParts[i].AS,pLen*sizeof(*(mat->AS)));
        memcpy(mat->JA + startOff,rowsParts[i].JA,pLen*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO REMOVE written nnz check manually
        for (uint i=0,w=0; i<mat->M; i++){
            if (mat->IRP[i] != w) 
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (j=0; j<conf->gridCols; j++){
                SPMATROW r = rowsParts[IDX2D(i,j,conf->gridCols)];
                for (uint jj=0; jj<r.len; jj++,w++){
                    if (mat->AS[w]!= r.AS[jj]){
                        ERRPRINT("MERGE ROW ERR AS\n"); return -1;}
                    if (mat->JA[w]!= r.JA[jj]){
                        ERRPRINT("MERGE ROW ERR JA\n"); return -1;}
                }
            }
        }
    }
    return EXIT_SUCCESS;
}

/*
 * merge @mat->M sparse rows @rows in sparse matrix @mat
 * EXPECTED @rows to be sorted in accord to trgt matrix row index @r
 * allocd arrays to hold non zero values and indexes into @mat
 */
static inline int mergeRows(SPMATROW* rows,spmat* mat){
    uint nzNum=0;
    //count nnz entries and alloc arrays for them
    for (uint r=0;   r<mat->M;   ++r){
        nzNum += rows[r].len;
        mat->IRP[r+1] = nzNum;
#ifdef ROWLENS
        mat->RL[r]  = rows[r].len
#endif
    }
    mat->NZ = nzNum;
    if (!(mat->AS = malloc(nzNum * sizeof(*(mat->AS))))){
        ERRPRINT("merged sparse matrix AS alloc errd\n");
        return EXIT_FAILURE;
    }  
    if (!(mat->JA = malloc(nzNum * sizeof(*(mat->JA))))){
        ERRPRINT("merged sparse matrix JA alloc errd\n");
        return EXIT_FAILURE;
    }
    ///POPOLATE WITH ROWS NNZ VALUES AND INDEXES
    for (uint r=0; r<mat->M; r++){
        //startOff+=rows[r].len;   //TODO row copy start offset alternative
        memcpy(mat->AS + mat->IRP[r],rows[r].AS,rows[r].len*sizeof(*(mat->AS)));
        memcpy(mat->JA + mat->IRP[r],rows[r].JA,rows[r].len*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO REMOVE written nnz check manually
        for (uint r=0,i=0; r<mat->M; r++){
            if (i != mat->IRP[r])
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (uint j=0; j<rows[r].len; j++,i++){
                if (mat->AS[i]!= rows[r].AS[j]){
                    ERRPRINT("MERGE ROW ERR AS\n"); return -1;}
                if (mat->JA[i]   != rows[r].JA[j]){
                    ERRPRINT("MERGE ROW ERR JA\n"); return -1;}
            }
        }
    }
        
    return EXIT_SUCCESS;
} 
    
spmat* spgemmGustavsonRowBlocks(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm rowBlocks of A, full B\tM=%u x N=%u\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT* accVect = NULL;
    SPMATROW*       accRows  = NULL;
    ///init AB matrix
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    //perform Gustavson over rows blocks -> M / @conf->gridRows
    uint rowBlock = AB->M/conf->gridRows, rowBlockRem = AB->M%conf->gridRows;
    ///aux structures alloc
    if (!(accVect = _initAccVectors(conf->gridRows,AB->N))){
        ERRPRINT("accVect init failed\n");
        goto _err;
    }
    if (!(accRows = calloc(AB->M , sizeof(*accRows)))) {
        ERRPRINT("accRows malloc failed\n");
        goto _err;
    }
    //init sparse row accumulators row indexes //TODO usefull in balance - reorder case
    for (uint r=0; r<AB->M; r++)    accRows[r].r = r;
    
    //TODO ADD PRIVATE -- REWATCH SHARED; ->  private(b,j,r)
    //TODO TODO #pragma omp parallel for schedule(static) shared(accRows,accVect,A)
    for (uint b=0,startRow=0,block=rowBlock+(rowBlockRem?1:0);
         b < conf->gridRows;
         startRow += block, block=rowBlock+(++b <rowBlockRem?1:0))//dist.unif.
    {
        /*TODO FOR SIMPLIFY -- DIRECT INDEXING
         *block = rowBlock + ( b < rowBlockRem=1:0 );
         *startRow = b*rowBlock + MIN(b,rowBlockRem)* (1);//add the extras for the most fair distribution
         *equivalent - more complex :   startRow = MIN(b,rowBlockRem)* (rowBlock+1) + MAX(b-rowBlock,0)*rowBlock;//direct compute 
         *startRow += block; // indirect compute -- cumulate
         */
        
        DEBUG   printf("block %u\t%u:%u(%u)\n",b,startRow,startRow+block-1,block);
        //row-by-row formulation in the given row block
        for (uint r=startRow;  r<startRow+block;  r++){
            //iterate over nz entry index c inside current row r
            for (uint c=A->IRP[r]; c<A->IRP[r+1]; c++) //row-by-row formul. accumulation
                scRowMul(A->AS[c], B, A->JA[c], accVect+b);
            //trasform accumulated dense vector to CSR with using the aux idxs
            if (sparsifyDenseVect(accVect+b,accRows + r,0)){
                fprintf(stderr,"sparsify failed ar row %u.\t aborting.\n",r);
                //TODO ABORT OPENMP COMPUTATION
                goto _err;
            }
            _resetAccVect(accVect+b);   //rezero for the next A row
        }
    }
    ///merge sparse row computed before
    if (mergeRows(accRows,AB))    goto _err;


    goto _free;
    
    _err:
    if(AB)  freeSpmat(AB);
    AB=NULL;    //nothing'll be returned
    _free:
    if (accRows){
        for (uint r=0; r<AB->M; r++)    freeSpRow(accRows + r);
        free(accRows);
    }
    if(accVect)     _freeAccVectors(accVect,conf->gridRows);

    return AB;
}

//PARTITIONS NOT ALLOCATED
spmat* spgemmGustavson2DBlocks(spmat* A,spmat* B, CONFIG* conf){ 
    DEBUG printf("spgemm A rowBlocks , B colBlocks\tM=%uxN=%u\n",A->M,B->N);
    uint* offsets = NULL;   //B group columns starting offset for each row
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
     
    //2D indexing aux vars
    uint gridSize=conf->gridRows*conf->gridCols, subRowsN=B->M*conf->gridCols;
    uint _rowBlock = AB->M/conf->gridRows, _rowBlockRem = AB->M%conf->gridRows;
    uint _colBlock = AB->N/conf->gridCols, _colBlockRem = AB->N%conf->gridCols;
    uint startRow,startCol,rowBlock,colBlock; //data division aux variables
    //aux vectors  
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPMATROW* accRowsParts=NULL, *accRowPart;
    ////get offsets for B column groups 
    //TODO UNIFORM SHARES OF COLS
    offsets = malloc( (subRowsN+1) * sizeof(*offsets) );
    if (!offsets)  {
        ERRPRINT("offsets malloc errd\n");
        goto _err;
    }
    ///OFFSETS COMPUTE FOR COL GROUPS -> O( B.NZ )
    for (uint r=0, j=0; r<B->M; j=B->IRP[++r]){
        for (uint gc=0,gcStartCol=0;  gc<conf->gridCols;  gc++){
            //goto the to the next  column group inside current row
            //contineing the B's nnz entries navigation (idx j)
            for (uint c=B->JA[j]; c<gcStartCol && j<B->IRP[r+1]; c=B->JA[++j]);
            offsets[ IDX2D(r,gc,conf->gridCols) ] = j;  //row's gc group startIdx
            gcStartCol += _colBlock + (gc < _colBlockRem ?1:0);//nextGroup start
        }
    }
    offsets[subRowsN] = B->NZ;
    ///AUX struct alloc
    //TODO aux vector of different sizes -> individual alloc later
    if (!(accVectors = calloc(gridSize, sizeof(*accVectors)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(accRowsParts = calloc(subRowsN, sizeof(*accRowsParts)))){
        ERRPRINT("accRowsParts calloc errd\n");
        goto _err;
    }

    uint bPartLen,bPartID,bPartOffset;//B partition acces aux vars
    //TODO OMP
    //TODO OMP HOW TO PARALLELIZE 2 FOR
    for (uint tileID = 0,t_i,t_j; tileID < gridSize; tileID++){
        ///get iteration's indexing variables
        //tile index in the 2D grid of AB computation TODO OMP HOW TO PARALLELIZE 2 FOR
        t_i = tileID/conf->gridCols;  //i-th row block
        t_j = tileID%conf->gridCols;  //j-th col block
        //get tile row-cols group FAIR sizes
        //rowBlock = _rowBlock + ( t_i < _rowBlockRem?1:0 ); 	        colBlock = _colBlock + ( t_j < _colBlockRem?1:0 );	        startRow = t_i * _rowBlock + MIN(t_i,_rowBlockRem)*(1);	        startCol = t_j * _colBlock + MIN(t_j,_colBlockRem)*(1);
        rowBlock = UNIF_REMINDER_DISTRI(t_i,_rowBlock,_rowBlockRem); 
        colBlock = UNIF_REMINDER_DISTRI(t_j,_colBlock,_colBlockRem);
        startRow = UNIF_REMINDER_DISTRI_STARTIDX(t_i,_rowBlock,_rowBlockRem);
        startCol = UNIF_REMINDER_DISTRI_STARTIDX(t_j,_colBlock,_colBlockRem);
        
        accV = accVectors + tileID; 
        if (_allocAuxVect(accV,colBlock))  goto _err;
         
        DEBUG{
            printf("rowBlock [%u\t%u:%u(%u)]\t",t_i,startRow,startRow+rowBlock-1,rowBlock);
            printf("colBlock [%u\t%u:%u(%u)]\n",t_j,startCol,startCol+colBlock-1,colBlock);
        }
        ///AB[t_i][t_j] block compute
        for (uint r=startRow;  r<startRow+rowBlock;  r++){
            //iterate over nz col index j inside current row r
            //row-by-row restricted to colsubset of B to get AB[r][:colBlock:]
            for (uint j=A->IRP[r],c; j<A->IRP[r+1]; j++){
                //get start of B[A->JA[j]][:colBlock:]
                c = A->JA[j]; // column of nnz entry in A[r][:] <-> target B row
                bPartID     = IDX2D(c,t_j,conf->gridCols); 
                bPartOffset = offsets[ bPartID ];
                bPartLen    = offsets[ bPartID + 1 ] - bPartOffset;

                scSparseVectMulPart(A->AS[j],B->AS+bPartOffset,
                  B->JA+bPartOffset,bPartLen,startCol,accV);
                //TODO SHIFT BACKWARD COLUMN INDEXES TO FIT THE ACCV ALLOCATED SPACE
            }

            accRowPart = accRowsParts + IDX2D(r,t_j,conf->gridCols);
            if (sparsifyDenseVect(accV,accRowPart,startCol)){
                fprintf(stderr,"err sparsify at block %u,%u\n",t_i,t_j);
                goto _err;
            }
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(accRowsParts,AB,conf))  goto _err;
    goto _free;

    _err:
    freeSpmat(AB);  AB = NULL; 
    _free:
    if (offsets)     free(offsets);
    if (accRowsParts){
        for(uint i=0;i<subRowsN;i++)    freeSpRow(accRowsParts+i);
        free(accRowsParts);
    }
    _freeAccVectorsChecks(accVectors,gridSize);
    
    return AB;
        
}
spmat* spgemmRowsGustavson2DAllocatedPartitions(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm rowBlocks of A, colBlocks of B\tM=%uxN=%u\n",A->M,B->N);
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
     

    _err:
    freeSpmat(AB);  AB = NULL; 
    _free:

    return AB;
        
}



spmat* sp3gemmGustavsonParallel(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    
    double end,start;
    start = omp_get_wtime();
   
    SPGEMM_INTERF computeSpGEMM = &spgemmGustavson2DBlocks; 
    //computeSpGEMM = spgemmRowsGustavson2DAllocatedPartitions;//TODO COMPARA CON VERSIONE CHE ALLOCA LE PARTIZIONI DELLE COLONNE
    //TODO ANY CONVENIENZA STORING B.COLPARTITIONS ... ALTRIMENTI:
    computeSpGEMM = &spgemmGustavsonRowBlocks;
    //alloc dense aux vector, reusable over 3 product 
    uint auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    //TODO PENSA BENE SE AGGIUNGERE RIUSO DI VETTORE DENSO NELLE 2 ALLOCAZIONE CON LA DIMENSIONE MASSIMA TRA LE 2 MATRICI.N........
    //TODO PRODOTTO IN 2
    
    spmat *RAC = NULL, *out = NULL;
    if (!(RAC = computeSpGEMM(R,AC,conf)))      goto _free;
#ifdef DEBUG_TEST_CBLAS
    DEBUG{ if(GEMMCheckCBLAS(R,AC,RAC))         goto _free; }
#endif
    if (!(out = computeSpGEMM(RAC,P,conf)))     goto _free;
#ifdef DEBUG_TEST_CBLAS
    DEBUG{ if(GEMMCheckCBLAS(RAC,P,out))        goto _free; }
#endif
     
    end = omp_get_wtime();
    VERBOSE 
    printf("spgemmRowsBasic 3 product of R:%ux%u AC:%ux%u P:%ux%u CSR sp.Mat, "
        "elapsed %lf\n",    R->M,R->N,AC->M,AC->N,P->M,P->N,end-start);

    _free:
    if (RAC)    freeSpmat(RAC);

    return out;

}
