//Developped by     Andrea Di Iorio - 0277550
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO quick hold few CSR cols partition sizes
#include <omp.h>

#include "SpGEMM.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

///inline exports
spmat*  allocSpMatrix(uint rows, uint cols);
int     allocSpMatrixInternal(uint rows, uint cols, spmat* mat);
spmat*  initSpMatrixSpGEMM(spmat* A, spmat* B);
int     reallocSpMatrix(spmat* mat,uint newSize);
void    freeSpmatInternal(spmat* mat);
void    freeSpmat(spmat* mat);

#ifdef DEBUG_TEST_CBLAS
    #pragma message("INCLUDING CBLAS")
    #include "SpGEMM_OMP_test.h"
#endif



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
static inline void scSparseRowMul(double scalar,spmat* mat,uint trgtR, THREAD_AUX_VECT* aux){
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


///OUTPUT SIZE PREDICTION
uint* spGEMMSizeUpperbound(spmat* A,spmat* B){
    AUDIT_INTERNAL_TIMES    Start = omp_get_wtime();
    uint* rowSizes = calloc((A->M+1), sizeof(*rowSizes));
    if (!rowSizes){
        ERRPRINT("spGEMMSizeUpperbound: rowSizes calloc errd\n");
        return NULL;
    }
    //#pragma omp parallel for schedule(static)
    for (uint r=0;  r<A->M; r++){
        for (uint jj=A->IRP[r],j,rlen;  jj<A->IRP[r+1]; jj++){
            j = A->JA[jj];
#ifdef ROWLENS
            rlen = B->RL[j];
#else
            rlen = B->IRP[j+1] - B->IRP[j];
#endif 
            rowSizes[r]     += rlen;
            rowSizes[A->M]  += rlen;    //TODO OMP REDUCTION SUM LIKE TO PARALLELIZE
        }
    }
    AUDIT_INTERNAL_TIMES    End= omp_get_wtime();
    VERBOSE 
        printf("spGEMMSizeUpperbound:%u\t%le s\n",rowSizes[A->M],End-Start);
    return rowSizes;
}

/*
 * check SpGEMM resulting matrix @AB = A * B nnz distribution in rows
 * with the preallocated, forecasted size in @forecastedSizes 
 * in @forecastedSizes there's for each row -> forecasted size 
 * and in the last entry the cumulative of the whole matrix
 */
static inline void checkOverallocPercent(uint* forecastedSizes,spmat* AB){
    for (int r=0,rSize,forecastedSize; (uint) r < AB->M; r++){
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
            printf("extra forecastedSize of row: %u\t=\t%lf %% \n",
              r,100*(forecastedSize-rSize) / (double) forecastedSize);
    }
    int extraMatrix = forecastedSizes[AB->M] - AB->NZ;
    VERBOSE
        printf("extra forecastedSize of the matrix: \t%u\t = %lf %% \n",
          extraMatrix, 100*extraMatrix /(double) forecastedSizes[AB->M]);
}

///SPARSE MATRIX PARTITIONING
/*
 * partition CSR sparse matrix @A in @gridCols columns partitions 
 * returning an offsets matrix out[i][j] = start of jth colPartition of row i
 * subdivide @A columns in uniform cols ranges in the output 
 */
uint* colsOffsetsPartitioningUnifRanges(spmat* A,uint gridCols){
    uint subRowsN = A->M * gridCols;
    uint _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols;
    uint* offsets = malloc( (subRowsN+1) * sizeof(*offsets) );
    if (!offsets)  {
        ERRPRINT("colsOffsetsPartitioningUnifRanges:\toffsets malloc errd\n");
        return NULL;
    }
    ///OFFSETS COMPUTE FOR COL GROUPS -> O( A.NZ )
    for (uint r=0, j=0;     r<A->M;     j=A->IRP[++r]){
        //navigate column groups inside current row
        for (uint gc=0,gcStartCol=0;  gc<gridCols;  gc++){
            //goto GroupCols start entry,keeping A's nnz entries navigation (idx j)
            //for (uint c=A->JA[j]; c<gcStartCol && j < A->IRP[r+1]; c=A->JA[++j]);
            while ( j < A->IRP[r+1] &&  A->JA[j] < gcStartCol )  j++;
            offsets[ IDX2D(r,gc,gridCols) ] = j;  //row's gc group startIdx
            gcStartCol += UNIF_REMINDER_DISTRI(gc,_colBlock,_colBlockRem);
        }
    }
    offsets[subRowsN] = A->NZ;
    return offsets;
}
/*
 * partition CSR sparse matrix @A in @gridCols columns partitions as 
 * indipended and allocated sparse matrixes and return them
 * subdivide @A columns in uniform cols ranges in the output 
 */
spmat* colsPartitioningUnifRanges(spmat* A,uint gridCols){
    spmat *colParts, *colPart;
    uint _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols, *tmpJA;
    double* tmpAS;
    ///alloc/init partitions structures
    if (!(colParts = calloc(gridCols, sizeof(*colParts)))){
        ERRPRINT("colsPartitioningUnifRanges\tcolumns partitions of A calloc fail\n");
        return NULL;
    }
    for (uint i=0,colBlock; i<gridCols; i++){
        colBlock = UNIF_REMINDER_DISTRI(i,_colBlock,_colBlockRem);
        if (allocSpMatrixInternal(A->M,colBlock,colParts + i)){
            ERRPRINT("colsPartitioningUnifRanges\tallocSpMatrixInternal partition err\n");
            goto _err;
        }
        //TODO overalloc A cols partitions NZ arrays, then realloc
        if (!((colParts+i)->AS = malloc(A->NZ * sizeof(*A->AS)))){
            ERRPRINT("colPart of A overalloc of AS errd\n");
            goto _err;
        }
        if (!((colParts+i)->JA = malloc(A->NZ * sizeof(*A->JA)))){
            ERRPRINT("colPart of A overalloc of JA errd\n");
            goto _err;
        }
    }
    //for each A col partition -> last copied nz index = nnz copied ammount
    uint* colPartsLens = alloca(gridCols * sizeof(colPartsLens));
    memset(colPartsLens, 0, sizeof(*colPartsLens) * gridCols);
    //OFFSET BASED COPY OF A.COL_GROUPS -> O( A.NZ )
    for (uint r=0, j=0;     r<A->M;     j=A->IRP[++r]){
        //navigate column groups inside current row
        for (uint gc=0,gcEndCol=0,i;  gc<gridCols ;  gc++,j+=i){
            i = 0;  //@i=len current subpartition of row @r to copy
            colPart = colParts + gc;
            colPart->IRP[r] = colPartsLens[gc];
            gcEndCol += UNIF_REMINDER_DISTRI(gc,_colBlock,_colBlockRem);
            //goto next GroupCols,keeping A's nnz entries navigation ( index j+i )
            //for (uint c=A->JA[j+i]; c<gcEndCol && j+i  < A->IRP[r+1]; c=A->JA[j+ ++i]);
            while ( j+i < A->IRP[r+1] && A->JA[j+i] < gcEndCol ) i++;
            memcpy(colPart->AS+colPart->IRP[r], A->AS+j, i*sizeof(*(A->AS)));
            memcpy(colPart->JA+colPart->IRP[r], A->JA+j, i*sizeof(*(A->JA)));
            
            colPartsLens[gc] += i;
#ifdef ROWLENS
            colPart->RL[r] = i;
#endif
        }
    }
    //TODO realloc overallcd A parts NZ arrays (TODO -> downsizing -> nofails?)
    for (uint i=0,partLen; i<gridCols; i++){
        colPart = colParts + i;
        partLen = colPartsLens[i];
        if (!(tmpAS = realloc(colPart->AS,partLen*sizeof(*(colPart->AS))))){
            ERRPRINT("realloc overallocated cols partition AS array\n");
            goto _err;
        }
        colPart->AS = tmpAS;
        if (!(tmpJA = realloc(colPart->JA,partLen*sizeof(*(colPart->JA))))){
            ERRPRINT("realloc overallocated cols partition JA array\n");
            goto _err;
        }
        colPart->JA         = tmpJA;
        colPart->NZ         = partLen;
        colPart->IRP[A->M]  = partLen;
    }
    return colParts;
    _err:
    for (uint i=0; i<gridCols; i++)   freeSpmatInternal(colParts+i);
    return NULL;
}

////partial [dense] results merging
///SpGEMM holder of accumulats 
SPGEMM_ACC* initSpGEMMAcc(uint entriesNum, uint accumulatorsNum){
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
  SPACC* accSparse, uint startColAcc){
    //sort nnz indexes of dense accumulator
    uint nnz = accV -> nnzIdxLast;
    sortuint(accV->nnzIdx,nnz); //sort nnz idx for ordered write
    accSparse -> len = nnz;
    uint accSparseStartIdx;
    accSparseStartIdx = __atomic_fetch_add(&(acc->lastAssigned),nnz,__ATOMIC_SEQ_CST); 
    /*#pragma omp atomic capture
    {   //fetch and add like .... 
        accSparseStartIdx = acc->lastAssigned;
        acc->lastAssigned += nnz;
    }*/
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
    for (uint i=0,j;    i<nnz;   i++){ 
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
    uint nzNum=0,j,rLen,idx,partsNum = mat->M * conf->gridCols;
    //TODO PARALLEL MERGE - SAVE STARTING OFFSET OF EACH PARTITION IN THE OUT MATRIX
    uint* rowsPartsOffsets=alloca(partsNum*sizeof(*rowsPartsOffsets));
    ///count nnz entries and alloc arrays for them
    for (uint r=0; r<mat->M; r++){
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
    //for (uint i=0,startOff=0,pLen;  i<partsNum;  startOff+=rowsParts[i++].len){
    //TODO PARALLEL PARTS MEMCOPY
    uint pLen; //omp for aux vars
    #pragma omp parallel for schedule(static) private(pLen)
    for (uint i=0;  i<partsNum; i++){
        pLen = rowsParts[i].len;
        memcpy(mat->AS + rowsPartsOffsets[i],rowsParts[i].AS,pLen*sizeof(*(mat->AS)));
        memcpy(mat->JA + rowsPartsOffsets[i],rowsParts[i].JA,pLen*sizeof(*(mat->JA)));
    }
    CONSISTENCY_CHECKS{ //TODO REMOVE written nnz check manually
        for (uint i=0,w=0; i<mat->M; i++){
            if (mat->IRP[i] != w) 
                {ERRPRINT("MERGE ROW ERR IRP\n");return -1;}
            for (j=0; j<conf->gridCols; j++){
                SPACC r = rowsParts[IDX2D(i,j,conf->gridCols)];
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
static inline int mergeRows(SPACC* rows,spmat* mat){
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
    //TODO PARALLEL COPY
    #pragma omp parallel for schedule(static)
    for (uint r=0; r<mat->M; r++){
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

///DENSE accumulator utils
/*
 * alloc threads' aux arrays once and split them in threads' structures
 * so free them once from the first thread struct, with the original pointers returned from the alloc
 */
static THREAD_AUX_VECT* _initAccVectors_monoalloc(uint num,uint size){ //TODO PERF WITH NEXT
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
//TODO free only allocated subvectors ... u.c. direct use _allocAuxVect
static void _freeAccVectorsChecks(THREAD_AUX_VECT* vectors,uint num){ 
    if (!vectors)   return;
    for (uint i=0; i<num; i++){
        if(vectors[i].v)        free(vectors[i].v);
        if(vectors[i].nnzIdx)   free(vectors[i].nnzIdx);
    }
    free(vectors);
}
static void freeAccVectors(THREAD_AUX_VECT* vectors,uint num){
    for (uint i=0; i<num; i++){
        free(vectors[i].v);
        free(vectors[i].nnzIdx);
    }
    free(vectors);
}


//////////////////////// COMPUTE CORE /////////////////////////////////////////
///1D
spmat* spgemmTemplate(spmat* A,spmat* B, CONFIG* conf){ //TODO TEMPLATE FOR AN SPGEMM
    DEBUG printf("spgemm\trows of A,\t(allcd) colBlocks of B\tM=%uxN=%u\n",A->M,B->N);
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;

    _err:
    freeSpmat(AB);  AB = NULL; 
    _free:

    return AB;
}
spmat* spgemmGustavsonRows(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm\trows of A,\tfull B\tM=%u x N=%u\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    uint* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
    
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(conf->threadNum,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;

    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(acc)
    for (uint r=0;  r<A->M; r++){    //row-by-row formulation
        //iterate over nz entry index c inside current row r
        acc = accVects + omp_get_thread_num();
        for (uint c=A->IRP[r]; c<A->IRP[r+1]; c++) //row-by-row formul
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
    if(accVects)    freeAccVectors(accVects,conf->threadNum);
    if(outAccumul)  freeSpGEMMAcc(outAccumul);

    return AB;
}

spmat* spgemmGustavsonRowBlocks(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm\trowBlocks of A,\tfull B\tM=%u x N=%u\n",A->M,B->N);
    ///thread aux
    THREAD_AUX_VECT *accVects = NULL,*acc;
    SPGEMM_ACC* outAccumul=NULL;
    uint* rowsSizes = NULL;
    ///init AB matrix with SPGEMM heuristic preallocation
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
    ///aux structures alloc 
    if (!(accVects = _initAccVectors(conf->gridRows,AB->N))){
        ERRPRINT("accVects init failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],AB->M)))  goto _err;
   
    //perform Gustavson over rows blocks -> M / @conf->gridRows
    uint rowBlock = AB->M/conf->gridRows, rowBlockRem = AB->M%conf->gridRows;
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint b,startRow,block; //omp for aux vars
    #pragma omp parallel for schedule(runtime) private(acc,startRow,block)
    for (b=0;   b < conf->gridRows; b++){
        block      = UNIF_REMINDER_DISTRI(b,rowBlock,rowBlockRem);
        startRow   = UNIF_REMINDER_DISTRI_STARTIDX(b,rowBlock,rowBlockRem);
       
        DEBUGPRINT{
            fflush(NULL);
            printf("block %u\t%u:%u(%u)\n",b,startRow,startRow+block-1,block);
            fflush(NULL);
        }
        //row-by-row formulation in the given row block
        for (uint r=startRow;  r<startRow+block;  r++){
            //iterate over nz entry index c inside current row r
            acc = accVects + b;
            for (uint c=A->IRP[r]; c<A->IRP[r+1]; c++) 
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
    if(accVects)    freeAccVectors(accVects,conf->gridRows);
    if(outAccumul)  freeSpGEMMAcc(outAccumul);

    return AB;
}

///2D
//PARTITIONS NOT ALLOCATED
spmat* spgemmGustavson2DBlocks(spmat* A,spmat* B, CONFIG* conf){ 
    DEBUG printf("spgemm\trowBlocks of A ,\tcolBlocks of B\tM=%uxN=%u\n",A->M,B->N);
    uint* offsets = NULL;   //B group columns starting offset for each row
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    SPGEMM_ACC* outAccumul=NULL;
    uint*   rowsSizes=NULL;
    if (!AB)    goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
     
    //2D indexing aux vars
    uint gridSize=conf->gridRows*conf->gridCols, subRowsN=B->M*conf->gridCols;
    uint _rowBlock = AB->M/conf->gridRows, _rowBlockRem = AB->M%conf->gridRows;
    uint _colBlock = AB->N/conf->gridCols, _colBlockRem = AB->N%conf->gridCols;
    uint startRow,startCol,rowBlock,colBlock; //data division aux variables
    //aux vectors  
    ////get offsets for B column groups 
    if (!(offsets = colsOffsetsPartitioningUnifRanges(B,conf->gridCols)))
        goto _err;

    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],subRowsN)))   goto _err;

    uint bPartLen,bPartID,bPartOffset;//B partition acces aux vars
    
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint tileID,t_i,t_j;                            //for aux vars
    #pragma omp parallel for schedule(runtime) \
      private(accV,accRowPart,rowBlock,colBlock,startRow,startCol,\
      bPartLen,bPartID,bPartOffset,t_i,t_j)
    for (tileID = 0; tileID < gridSize; tileID++){
        ///get iteration's indexing variables
        //tile index in the 2D grid of AB computation TODO OMP HOW TO PARALLELIZE 2 FOR
        t_i = tileID/conf->gridCols;  //i-th row block
        t_j = tileID%conf->gridCols;  //j-th col block
        //get tile row-cols group FAIR sizes
        rowBlock = UNIF_REMINDER_DISTRI(t_i,_rowBlock,_rowBlockRem); 
        startRow = UNIF_REMINDER_DISTRI_STARTIDX(t_i,_rowBlock,_rowBlockRem);
        startCol = UNIF_REMINDER_DISTRI_STARTIDX(t_j,_colBlock,_colBlockRem);
        
        accV = accVectors + tileID; 
         
        DEBUGPRINT{
            fflush(NULL);
            colBlock = UNIF_REMINDER_DISTRI(t_j,_colBlock,_colBlockRem);
            printf("rowBlock [%u\t%u:%u(%u)]\t",t_i,startRow,startRow+rowBlock-1,rowBlock);
            printf("colBlock [%u\t%u:%u(%u)]\n",t_j,startCol,startCol+colBlock-1,colBlock);
            fflush(NULL);
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
            }

            accRowPart = outAccumul->accs + IDX2D(r,t_j,conf->gridCols);
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,conf))  goto _err;
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




spmat* spgemmGustavson2DBlocksAllocated(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm\trowBlocks of A,\tcolBlocks (allcd) of B\tM=%uxN=%u\n",A->M,B->N);
    spmat *AB = NULL, *colPartsB = NULL, *colPart;
    uint*   rowsSizes=NULL;
    //aux vectors  
    SPGEMM_ACC* outAccumul=NULL;
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPACC* accRowPart;
    if (!(AB = allocSpMatrix(A->M,B->N)))           goto _err;
    if (!(rowsSizes = spGEMMSizeUpperbound(A,B)))   goto _err;
    //2D indexing aux vars
    uint gridSize=conf->gridRows*conf->gridCols, subRowsN=B->M*conf->gridCols;
    uint _rowBlock = AB->M/conf->gridRows, _rowBlockRem = AB->M%conf->gridRows;
    uint _colBlock = AB->N/conf->gridCols, _colBlockRem = AB->N%conf->gridCols;
    uint startRow,startCol,rowBlock,colBlock; //data division aux variables
    ////B cols  partition in CSRs
    if (!(colPartsB = colsPartitioningUnifRanges(B,conf->gridCols)))  goto _err;
    ///other AUX struct alloc
    if (!(accVectors = _initAccVectors(gridSize,_colBlock+(_colBlockRem?1:0)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(outAccumul = initSpGEMMAcc(rowsSizes[AB->M],subRowsN)))   goto _err;
    
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint tileID,t_i,t_j;    //for aux vars
    #pragma omp parallel for schedule(runtime) \
      private(accV,accRowPart,colPart,rowBlock,colBlock,startRow,startCol,t_i,t_j)
    for (tileID = 0; tileID < gridSize; tileID++){
        ///get iteration's indexing variables
        //tile index in the 2D grid of AB computation TODO OMP HOW TO PARALLELIZE 2 FOR
        t_i = tileID/conf->gridCols;  //i-th row block
        t_j = tileID%conf->gridCols;  //j-th col block
        //get tile row-cols group FAIR sizes
        rowBlock = UNIF_REMINDER_DISTRI(t_i,_rowBlock,_rowBlockRem); 
        colBlock = UNIF_REMINDER_DISTRI(t_j,_colBlock,_colBlockRem);
        startRow = UNIF_REMINDER_DISTRI_STARTIDX(t_i,_rowBlock,_rowBlockRem);
        startCol = UNIF_REMINDER_DISTRI_STARTIDX(t_j,_colBlock,_colBlockRem);
        
        colPart = colPartsB + t_j;
        accV = accVectors + tileID; 
         
        DEBUGPRINT{
            fflush(NULL);
            printf("rowBlock [%u\t%u:%u(%u)]\t",t_i,startRow,startRow+rowBlock-1,rowBlock);
            printf("colBlock [%u\t%u:%u(%u)]\n",t_j,startCol,startCol+colBlock-1,colBlock);
            fflush(NULL);
        }
        ///AB[t_i][t_j] block compute
        for (uint r=startRow;  r<startRow+rowBlock;  r++){
            //iterate over nz col index j inside current row r
            //row-by-row restricted to colsubset of B to get AB[r][:colBlock:]
            for (uint j=A->IRP[r],c,bRowStart,bRowLen; j<A->IRP[r+1]; j++){
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

            accRowPart = outAccumul->accs + IDX2D(r,t_j,conf->gridCols);
            sparsifyDenseVect(outAccumul,accV,accRowPart,startCol);
            _resetAccVect(accV);
        }
    }
    if (mergeRowsPartitions(outAccumul->accs,AB,conf))  goto _err;
    AUDIT_INTERNAL_TIMES	End=omp_get_wtime();
    DEBUG                   checkOverallocPercent(rowsSizes,AB);
    goto _free;

    _err:
    ERRPRINT("spgemmGustavson2DBlocksAllocated failed\n");
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    if (colPartsB){
        for (uint i=0; i<conf->gridCols; i++)   freeSpmatInternal(colPartsB+i);
        free(colPartsB);
    }
    if (rowsSizes)   free(rowsSizes);
    if (accVectors)  freeAccVectors(accVectors,gridSize);
    if (outAccumul)  freeSpGEMMAcc(outAccumul);
    
    return AB;
        
}



spmat* sp3gemmGustavsonParallel(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    
    double end,start,partial,flops;
    start = omp_get_wtime();
   
    SPGEMM_INTERF computeSpGEMM = (SPGEMM_INTERF) conf->spgemmFunc;
    if (!computeSpGEMM){
        //TODO runtime decide witch spgemm implementation to use if not given
        computeSpGEMM = &spgemmGustavson2DBlocks; 
        //computeSpGEMM = &spgemmGustavson2DBlocksAllocated;//TODO COMPARA CON VERSIONE CHE ALLOCA LE PARTIZIONI DELLE COLONNE
        //TODO ANY CONVENIENZA STORING B.COLPARTITIONS ... ALTRIMENTI:
        //computeSpGEMM = &spgemmGustavsonRowBlocks;
    }
    //alloc dense aux vector, reusable over 3 product 
    uint auxVectSize = MAX(R->N,AC->N);
    auxVectSize      = MAX(auxVectSize,P->N);
    //TODO arrays sovrallocati per poter essere riusati nelle 2 SpGEMM
    
    spmat *RAC = NULL, *out = NULL;
    if (!(RAC = computeSpGEMM(R,AC,conf)))      goto _free;
#ifdef DEBUG_TEST_CBLAS
    if(GEMMCheckCBLAS(R,AC,RAC))                goto _free;
#endif
    AUDIT_INTERNAL_TIMES    partial = End - Start;
    if (!(out = computeSpGEMM(RAC,P,conf)))     goto _free;
#ifdef DEBUG_TEST_CBLAS
    if(GEMMCheckCBLAS(RAC,P,out))               goto _free;
#endif
    
    ///time accounting and prints 
    end = omp_get_wtime();
    Elapsed         = end - start;
    ElapsedInternal = End - Start + partial;
    flops = ( 2 * R->NZ * P->NZ * AC->NZ ) / ( Elapsed );
    DEBUG 
      printf("sp3gemmGustavsonParallel of R:%ux%u AC:%ux%u P:%ux%u CSR sp.Mat",
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
