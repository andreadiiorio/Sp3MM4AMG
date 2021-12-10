//Developped by     Andrea Di Iorio - 0277550
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO quick hold few CSR cols partition sizes
#include <omp.h>

#include "SpGEMM.h"
#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"


//global vars	->	audit
double Start,End,Elapsed,ElapsedInternal;

////AUX FUNCTIONS
///scalar-vector multiply

//TODO che guadagno si ha ad utilizzare solo la versione generica delle successive 2 funzioni
/*
 * Sparse vector part <->scalar multiplication in a dense output
 * sparse vector part will hold nnz values in @vectVals 
 * with corresponding indexes in @vectIdxs in range [0,@vectLen]
 * resulting vector accumulated in a dense array in @aux->v, along with nnzIdx
 * both of accumulator's dense array and nnzIdx in @aux and has to be big @vectLen
 */
static inline void scSparseVectMul(double scalar,
  double* vectVals,ulong* vectIdxs,ulong vectLen, THREAD_AUX_VECT* aux){
    for (ulong i=0,j; i<vectLen; i++){
        j = vectIdxs[i];
        DEBUGCHECKS{
            if (j>aux->vLen){
                fprintf(stderr,"index %lu outside vLen %lu\n",j,aux->vLen);
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
  ulong* vectIdxs,ulong vectLen,ulong startIdx,THREAD_AUX_VECT* aux){
    for (ulong i=0,j; i<vectLen; i++){
        j = vectIdxs[i] - startIdx; //shift back nnz value index for the accumul
        DEBUGCHECKS{ //TODO REMOVE
            if (j>aux->vLen){
                fprintf(stderr,"index %lu outside vLen %lu\n",j,aux->vLen);
                exit(1);
            }
        }
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))   aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += vectVals[i] * scalar;  //accumulate
    }
}

/////////////SIMD - REDUCTION version of aboves
#define S_VECT_PROD_SIMD_NO_BRANCH  //TODO ADVICE ASK
static inline void scSparseVectMulReduction(double scalar,
  double* vectVals,ulong* vectIdxs,ulong vectLen, THREAD_AUX_VECT* aux){
    double* v = aux->v;
    //#pragma omp parallel for reduction (+:v[:vectLen])
    #pragma omp simd reduction (+:v[:vectLen])
    for (ulong i=0; i<vectLen; i++){
        ulong j = vectIdxs[i];
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        #ifndef S_VECT_PROD_SIMD_NO_BRANCH
        if (!v[j])    aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        #endif
        v[j] += vectVals[i] * scalar;  //accumulate
    }
    #ifdef S_VECT_PROD_SIMD_NO_BRANCH
    for(ulong i=0; i<vectLen; i++){
        if(v[i])   aux->nnzIdx[ aux->nnzIdxLast++ ] = i;
    }
    #endif
}
static inline void scSparseVectMulPartReduction(double scalar,double* vectVals,
  ulong* vectIdxs,ulong vectLen,ulong startIdx,THREAD_AUX_VECT* aux){
    double* v = aux->v;
    //#pragma omp parallel for reduction (+:v[:vectLen])
    #pragma omp simd reduction (+:v[:vectLen])
    for (ulong i=0; i<vectLen; i++){
        ulong j = vectIdxs[i] - startIdx; //shift back nnz value index for the accumul
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        #ifndef S_VECT_PROD_SIMD_NO_BRANCH
        if (!v[j])   aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        #endif
        v[j] += vectVals[i] * scalar;  //accumulate
    }
    #ifdef S_VECT_PROD_SIMD_NO_BRANCH
    for(ulong i=0; i<vectLen; i++){
        if(v[i])   aux->nnzIdx[ aux->nnzIdxLast++ ] = i;
    }
    #endif
}
/////////////
///TODO OLD DIRECT USE OF SCALAR<->ROW MULTIPLICATION -- REMOVABLE
static inline void _scRowMul(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux){
    for (ulong c=mat->IRP[trgtR],j;  c<mat->IRP[trgtR+1];  c++){//scanning trgtRow
        j = mat->JA[c];
        //append new nonzero index to auxVNNZeroIdxs for quick sparsify
        if (!(aux->v[j]))    aux->nnzIdx[ aux->nnzIdxLast++ ] = j;
        aux->v[j] += mat->AS[c] * scalar;  //accumulate
    }
}
///TODO IMPLEMENT SCALAR<->ROW MUL AS GENERIC SPARSE VECTOR<->SCALAR MUL
static inline void scSparseRowMul(double scalar,spmat* mat,ulong trgtR, THREAD_AUX_VECT* aux){
    ulong  rowStartIdx = mat->IRP[trgtR],rowLen;
#ifdef ROWLENS
    rowLen = mat->RL[trgtR];
#else
    rowLen = mat->IRP[trgtR+1]-rowStartIdx;
#endif
    scSparseVectMul(scalar,mat->AS+rowStartIdx,mat->JA+rowStartIdx,rowLen,aux);
    //scSparseVectMulPart(scalar,mat->AS+rowStartIdx,mat->JA+rowStartIdx,rowLen,0,aux);
    //TODO TODO check impact of generic version use
}


///OUTPUT SIZE PREDICTION
ulong* spGEMMSizeUpperbound(spmat* A,spmat* B){
    AUDIT_INTERNAL_TIMES    Start = omp_get_wtime();
    ulong* rowSizes = calloc((A->M+1), sizeof(*rowSizes));
    if (!rowSizes){
        ERRPRINT("spGEMMSizeUpperbound: rowSizes calloc errd\n");
        return NULL;
    }
    ulong fullMatBound = 0;
    #pragma omp parallel for schedule(static) reduction(+:fullMatBound)
    for (ulong r=0;  r<A->M; r++){
        for (ulong jj=A->IRP[r],j,rlen;  jj<A->IRP[r+1]; jj++){
            j = A->JA[jj];
#ifdef ROWLENS
            rlen = B->RL[j];
#else
            rlen = B->IRP[j+1] - B->IRP[j];
#endif 
            rowSizes[r]     += rlen;
            fullMatBound    += rlen;
            //rowSizes[A->M]  += rlen;    //TODO OMP REDUCTION SUM LIKE TO PARALLELIZE
        }
    }
    rowSizes[A->M]  = fullMatBound;
    AUDIT_INTERNAL_TIMES    End= omp_get_wtime();
    VERBOSE 
        printf("spGEMMSizeUpperbound:%lu\t%le s\n",rowSizes[A->M],End-Start);
    return rowSizes;
}

/*
 * check SpGEMM resulting matrix @AB = A * B nnz distribution in rows
 * with the preallocated, forecasted size in @forecastedSizes 
 * in @forecastedSizes there's for each row -> forecasted size 
 * and in the last entry the cumulative of the whole matrix
 */
static inline void checkOverallocPercent(ulong* forecastedSizes,spmat* AB){
    for (ulong r=0,rSize,forecastedSize; r < AB->M; r++){
        forecastedSize = forecastedSizes[r];
#ifdef ROWLENS
        rSize = AB->RL[r];
#else
        rSize = AB->IRP[r+1] - AB->IRP[r];
#endif
        DEBUGCHECKS{
            if ( forecastedSize < rSize ){
                ERRPRINT("BAD FORECASTING\n");
                exit(22);
            }
        }
        DEBUGPRINT
            printf("extra forecastedSize of row: %lu\t=\t%lf %% \n",
              r,100*(forecastedSize-rSize) / (double) forecastedSize);
    }
    ulong extraMatrix = forecastedSizes[AB->M] - AB->NZ;
    VERBOSE
        printf("extra forecastedSize of the matrix: \t%lu\t = %lf %% \n",
          extraMatrix, 100*extraMatrix /(double) forecastedSizes[AB->M]);
}


////partial [dense] results merging
///SpGEMM holder of accumulats 
SPGEMM_ACC* initSpGEMMAcc(ulong entriesNum, ulong accumulatorsNum){
    SPGEMM_ACC* out = calloc(1,sizeof(*out));
    if (!out){
        ERRPRINT("initSpGEMMAcc:    out calloc errd\n");
        return NULL;
    }
    out->size = entriesNum;
    if (!(out->JA = malloc(entriesNum * sizeof(*(out->JA))))){
        ERRPRINT("initSpGEMMAcc:    JA malloc errd\n");
        goto _err;
    }
    if (!(out->AS = malloc(entriesNum * sizeof(*(out->AS))))){
        ERRPRINT("initSpGEMMAcc:    AS malloc errd\n");
        goto _err;
    }
    if (!(out->accs = malloc(accumulatorsNum * sizeof(*(out->accs))))){
        ERRPRINT("initSpGEMMAcc:    accs malloc errd\n");
        goto _err;
    }
    return out;

    _err:
    if (out->JA)    free(out->JA);
    if (out->AS)    free(out->AS);
    if (out->accs)  free(out->accs);
    if (out)        free(out);
    return NULL;
}

static inline void freeSpGEMMAcc(SPGEMM_ACC* acc){
    free(acc->JA);
    free(acc->AS);
    free(acc->accs);
    free(acc);
}

/*
 * sparsify dense accumulated vector @accV (with shifted of @startColAcc) 
 * into sparse accumulator @accSparse that'll use space for nnz entries from @acc
*/
static inline void sparsifyDenseVect(SPGEMM_ACC* acc,THREAD_AUX_VECT* accV,
  SPACC* accSparse, ulong startColAcc){
    //sort nnz indexes of dense accumulator
    ulong nnz = accV -> nnzIdxLast;
    sortulong(accV->nnzIdx,nnz); //sort nnz idx for ordered write
    accSparse -> len = nnz;
    ulong accSparseStartIdx;
    //accSparseStartIdx = __atomic_fetch_add(&(acc->lastAssigned),nnz,__ATOMIC_ACQ_REL); 
    #pragma omp atomic capture
    {   //fetch and add like .... 
        accSparseStartIdx = acc->lastAssigned;
        acc->lastAssigned += nnz;
    }
    DEBUGCHECKS{
        if (acc->lastAssigned >= acc->size){
            ERRPRINT("OMP ATOMIC OR SG ERRD IN SPACE ASSIGNMENTS...\n");
            exit(22);
        }
    }
    //
    accSparse -> AS = acc->AS + accSparseStartIdx; 
    accSparse -> JA = acc->JA + accSparseStartIdx; 
    ///sparsify dense acc.v into row
    for (ulong i=0,j;    i<nnz;   i++){ 
        j = accV -> nnzIdx[i];        //shifted idx of a nnz of sp.Vect accumulator
        accSparse -> JA[i] = j + startColAcc;
        accSparse -> AS[i] = accV->v[j];
    }
    //TODO USED TO BE A CONSISTENCY CHECK
}

/*
 * merge @conf->gridCols*@mat->M sparse rows partitions into @mat
 * EXPECTED rowsParts @rowsParts to be sorted in accord to the 
 * 2D rowMajor computing grid given in @conf
 * allocd arrays to hold non zero values and indexes into @mat
 */
static inline int mergeRowsPartitions(SPACC* rowsParts,spmat* mat,
  CONFIG* conf){
    ulong nzNum=0,j,rLen,idx,partsNum = mat->M * conf->gridCols;
    //TODO PARALLEL MERGE - SAVE STARTING OFFSET OF EACH PARTITION IN THE OUT MATRIX
    ulong* rowsPartsOffsets=alloca(partsNum*sizeof(*rowsPartsOffsets));
    ///count nnz entries and alloc arrays for them
    for (ulong r=0; r<mat->M; r++){
        //for each partition ->get len -> outMat.IRP and aux offsets  
        for (j=0,rLen=0; j<conf->gridCols; j++){
            idx = IDX2D(r,j,conf->gridCols);
            rowsPartsOffsets[idx]=nzNum+rLen;//part start=prev accumulated end
            rLen += rowsParts[idx].len;
        }
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
    //OLD FOR
    //for (ulong i=0,startOff=0,pLen;  i<partsNum;  startOff+=rowsParts[i++].len){
    //TODO PARALLEL PARTS MEMCOPY
    ulong pLen; //omp for aux vars
    #pragma omp parallel for schedule(static) private(pLen)
    for (ulong i=0;  i<partsNum; i++){
        pLen = rowsParts[i].len;
        memcpy(mat->AS + rowsPartsOffsets[i],rowsParts[i].AS,pLen*sizeof(*(mat->AS)));
        memcpy(mat->JA + rowsPartsOffsets[i],rowsParts[i].JA,pLen*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO REMOVE written nnz check manually
        for (ulong i=0,w=0; i<mat->M; i++){
            if (mat->IRP[i] != w) 
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (j=0; j<conf->gridCols; j++){
                SPACC r = rowsParts[IDX2D(i,j,conf->gridCols)];
                for (ulong jj=0; jj<r.len; jj++,w++){
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
static inline int mergeRows(SPACC* rows,spmat* mat){
    ulong nzNum=0;
    //count nnz entries and alloc arrays for them
    for (ulong r=0;   r<mat->M;   ++r){
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
    //TODO PARALLEL COPY
    #pragma omp parallel for schedule(static)
    for (ulong r=0; r<mat->M; r++){
        memcpy(mat->AS + mat->IRP[r],rows[r].AS,rows[r].len*sizeof(*(mat->AS)));
        memcpy(mat->JA + mat->IRP[r],rows[r].JA,rows[r].len*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO REMOVE written nnz check manually
        for (ulong r=0,i=0; r<mat->M; r++){
            if (i != mat->IRP[r])
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (ulong j=0; j<rows[r].len; j++,i++){
                if (mat->AS[i]!= rows[r].AS[j]){
                    ERRPRINT("MERGE ROW ERR AS\n"); return -1;}
                if (mat->JA[i]   != rows[r].JA[j]){
                    ERRPRINT("MERGE ROW ERR JA\n"); return -1;}
            }
        }
    }
    return EXIT_SUCCESS;
} 

///DENSE accumulator utils
/*
 * alloc threads' aux arrays once and split them in threads' structures
 * so free them once from the first thread struct, with the original pointers returned from the alloc
 */
static THREAD_AUX_VECT* _initAccVectors_monoalloc(ulong num,ulong size){ //TODO PERF WITH NEXT
    THREAD_AUX_VECT* out    = NULL;
    double* vAll            = NULL;
    ulong* vAllNzIdx         = NULL;
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
    for (ulong i=0; i<num; i++){
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
static inline int _allocAuxVect(THREAD_AUX_VECT* v,ulong size){
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
static THREAD_AUX_VECT* _initAccVectors(ulong num,ulong size){
    THREAD_AUX_VECT* out    = NULL;
    if (!(out = calloc(num,sizeof(*out)))){
        ERRPRINT("_initAccVectors aux struct alloc failed\n");
        return NULL;
    }
    for (ulong i=0; i<num; i++){
        if (_allocAuxVect(out+i,size))   goto _err;
    }
    return out;
    
    _err:
    for (ulong i=0; i<num; i++){
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
//TODO free only allocated subvectors ... u.c. direct use _allocAuxVect
static void _freeAccVectorsChecks(THREAD_AUX_VECT* vectors,ulong num){ 
    if (!vectors)   return;
    for (ulong i=0; i<num; i++){
        if(vectors[i].v)        free(vectors[i].v);
        if(vectors[i].nnzIdx)   free(vectors[i].nnzIdx);
    }
    free(vectors);
}
static void freeAccVectors(THREAD_AUX_VECT* vectors,ulong num){
    for (ulong i=0; i<num; i++){
        free(vectors[i].v);
        free(vectors[i].nnzIdx);
    }
    free(vectors);
}


//////////////////////// COMPUTE CORE /////////////////////////////////////////
///1D
spmat* spgemmTemplate(spmat* A,spmat* B, CONFIG* cfg){ //TODO TEMPLATE FOR AN SPGEMM
    DEBUG printf("spgemm\trows of A,\t(allcd) colBlocks of B\tM=%luxN=%lu\n",A->M,B->N);
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;

    _err:
    freeSpmat(AB);  AB = NULL; 
    _free:

    return AB;
}
spmat* spgemmRowByRow(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trows of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    ulong* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
    
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(cfg->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;

    ((CHUNKS_DISTR_INTERF) cfg->chunkDistrbFunc) (AB->M,AB,cfg);
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(acc)
    for (ulong r=0;  r<A->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        acc = accVects + omp_get_thread_num();
        for (ulong c=A->IRP[r]; c<A->IRP[r+1]; c++) //row-by-row formul
            scSparseRowMul(A->AS[c], B, A->JA[c], acc);
        //trasform accumulated dense vector to a CSR row
        sparsifyDenseVect(outAccumul,acc,outAccumul->accs + r,0);
        _resetAccVect(acc);   //rezero for the next A row
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,AB))    goto _err;
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

spmat* spgemmRowByRow1DBlocks(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trowBlocks of A,\tfull B\tM=%lu x N=%lu\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    ulong* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
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
            for (ulong c=A->IRP[r]; c<A->IRP[r+1]; c++) 
                scSparseRowMul(A->AS[c], B, A->JA[c], acc);
            //trasform accumulated dense vector to a CSR row
            sparsifyDenseVect(outAccumul,acc,outAccumul->accs + r,0);
            _resetAccVect(acc);   //rezero for the next A row
        }
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,AB))    goto _err;
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
spmat* spgemmRowByRow2DBlocks(spmat* A,spmat* B, CONFIG* cfg){ 
    DEBUG printf("spgemm\trowBlocks of A ,\tcolBlocks of B\tM=%luxN=%lu\n",A->M,B->N);
    ulong* offsets = NULL;   //B group columns starting offset for each row
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    SPGEMM_ACC* outAccumul=NULL;
    ulong*   rowsSizes=NULL;
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
     
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, subRowsN=B->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    //aux vectors  
    ////get offsets for B column groups 
    if (!(offsets = colsOffsetsPartitioningUnifRanges(B,cfg->gridCols)))
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
            for (ulong j=A->IRP[r],c; j<A->IRP[r+1]; j++){
                //get start of B[A->JA[j]][:colBlock:]
                c = A->JA[j]; // column of nnz entry in A[r][:] <-> target B row
                bPartID     = IDX2D(c,t_j,cfg->gridCols); 
                bPartOffset = offsets[ bPartID ];
                bPartLen    = offsets[ bPartID + 1 ] - bPartOffset;

                scSparseVectMulPart(A->AS[j],B->AS+bPartOffset,
                  B->JA+bPartOffset,bPartLen,startCol,accV);
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,cfg->gridCols);
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
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

spmat* spgemmRowByRow2DBlocksAllocated(spmat* A,spmat* B, CONFIG* cfg){
    DEBUG printf("spgemm\trowBlocks of A,\tcolBlocks (allcd) of B\tM=%luxN=%lu\n",A->M,B->N);
    spmat *AB = NULL, *colPartsB = NULL, *colPart;
    ulong*   rowsSizes=NULL;
    //aux vectors  
    SPGEMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    if (!(AB = allocSpMatrix(A->M,B->N)))           goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
    //2D indexing aux vars
    ulong gridSize=cfg->gridRows*cfg->gridCols, subRowsN=B->M*cfg->gridCols;
    ulong _rowBlock = AB->M/cfg->gridRows, _rowBlockRem = AB->M%cfg->gridRows;
    ulong _colBlock = AB->N/cfg->gridCols, _colBlockRem = AB->N%cfg->gridCols;
    ulong startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////B cols  partition in CSRs
    if (!(colPartsB = colsPartitioningUnifRanges(B,cfg->gridCols)))  goto _err;
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
            for (ulong j=A->IRP[r],c,bRowStart,bRowLen; j<A->IRP[r+1]; j++){
                //get start of B[A->JA[j]][:colBlock:]
                c = A->JA[j]; // column of nnz entry in A[r][:] <-> target B row
                //scSparseRowMul(A->AS[j],colPart,c,accV);//TODO GENERIC VERSION USEFUL
                bRowStart = colPart->IRP[c];
#ifdef ROWLENS
                bRowLen   = colPart->RL[c];
#else
                bRowLen   = colPart->IRP[c+1] - bRowStart;
#endif
                scSparseVectMulPart(A->AS[j],colPart->AS+bRowStart,colPart->JA+bRowStart,
                    bRowLen,startCol,accV);
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,cfg->gridCols);
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,cfg))  goto _err;
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
///SP3GEMM
spmat* sp3gemmRowByRowPair(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPGEMM_INTERF spgemm){
    
    double end,start,partial,flops;
    start = omp_get_wtime();
   
    if (!spgemm){
        //TODO runtime on sizes decide witch spgemm implementation to use if not given
        spgemm = &spgemmRowByRow2DBlocks; 
    }
    /* TODO 
    alloc dense aux vector, reusable over 3 product 
    TODO arrays sovrallocati per poter essere riusati nelle 2 SpGEMM
    ulong auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    */
    spmat *RAC = NULL, *out = NULL;
    if (!(RAC = spgemm(R,AC,cfg)))      goto _free;
    AUDIT_INTERNAL_TIMES    partial = End - Start;
    if (!(out = spgemm(RAC,P,cfg)))     goto _free;
    
    ///time accounting and prints 
    end = omp_get_wtime();
    Elapsed         = end - start;
    ElapsedInternal = End - Start + partial;
    flops = ( 2 * R->NZ * P->NZ * AC->NZ ) / ( Elapsed );
    DEBUG 
      printf("sp3gemmGustavsonParallel of R:%lux%lu AC:%lux%lu P:%lux%lu CSR sp.Mat",
        R->M,R->N,AC->M,AC->N,P->M,P->N);
    VERBOSE {
        printf("elapsed %le - flops %le",Elapsed,flops);
        AUDIT_INTERNAL_TIMES    printf("\tinternalTime: %le",ElapsedInternal);
        printf("\n");
    }
    _free:
    if (RAC)    freeSpmat(RAC);

    return out;
}
spmat* sp3gemmRowByRowMerged(spmat* R,spmat* AC,spmat* P,CONFIG* cfg,SPGEMM_INTERF spgemm){
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
    if (!(rowsSizes = spGEMMSizeUpperbound(R,AC)))   goto _err;
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
    /* TODO PARALLELISM ALTERNATIVES:
       ???
        il carico per thread puo essere molto variabile -> dinamic(folded)  
    */ 
    #pragma omp parallel for schedule(runtime) private(accRAC,accRACP,c)
    for (ulong r=0;  r<R->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        accRAC  = accVectorsR_AC  + omp_get_thread_num();
        accRACP = accVectorsRAC_P + omp_get_thread_num();
        for (ulong j=R->IRP[r]; j<R->IRP[r+1]; j++) //row-by-row formul
            scSparseRowMul(R->AS[j], AC, R->JA[j], accRAC);
        //forward the computed R*AC row (in the dense, indexed acc) for P compute
        for (ulong j=0; j<accRAC->nnzIdxLast; j++){
            c = accRAC->nnzIdx[j];    
            scSparseRowMul(accRAC->v[c],P,c,accRACP);
        }
        //trasform accumulated dense vector to a CSR row
        sparsifyDenseVect(outAccumul,accRACP,outAccumul->accs+r,0);
        _resetAccVect(accRAC);
        _resetAccVect(accRACP);
    }
    ///merge sparse row computed before
    if (mergeRows(outAccumul->accs,out))    goto _err;
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
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

