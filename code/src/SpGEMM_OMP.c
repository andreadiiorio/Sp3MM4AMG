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

#ifdef DEBUG_TEST_CBLAS
    #include "SpGEMM_OMP_test.h"
#endif

//inline export here 
spmat* allocSpMatrix(uint rows, uint cols);
int allocSpMatrixInternal(uint rows, uint cols, spmat* mat);
void freeSpmatInternal(spmat* mat);

//global vars	-	audit
double Start;

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


///AUX
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
        ERRPRINT("offsets malloc errd\n");
        return NULL;
    }
    ///OFFSETS COMPUTE FOR COL GROUPS -> O( B.NZ )
    for (uint r=0, j=0; r<A->M; j=A->IRP[++r]){
        for (uint gc=0,gcStartCol=0;  gc<gridCols;  gc++){
            //goto the to the next  column group inside current row
            //continuing the B's nnz entries navigation (idx j)
            for (uint c=A->JA[j]; c<gcStartCol && j<A->IRP[r+1]; c=A->JA[++j]);
            offsets[ IDX2D(r,gc,gridCols) ] = j;  //row's gc group startIdx
            //gcStartCol += _colBlock + (gc < _colBlockRem ?1:0);//nextGroup start
            gcStartCol += UNIF_REMINDER_DISTRI(gc,_colBlock,_colBlockRem);
        }
    }
    offsets[subRowsN] = A->NZ;
    return offsets;
}
/*
 * partition CSR sparse matrix @A in @gridCols columns partitions as indipended
 * sparse matrixes and return them
 * subdivide @A columns in uniform cols ranges in the output 
 */
spmat* colsPartitioningUnifRanges(spmat* A,uint gridCols){
    spmat *colParts, *colPart;
    uint _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols, *tmpJA;
    double* tmpAS;
    //alloc/init partitions structures
    if (!(colParts = calloc(gridCols, sizeof(*colParts)))){
        ERRPRINT("columns partitions of A calloc fail\n");
        return NULL;
    }
    for (uint i=0,colBlock; i<gridCols; i++){
        colBlock = UNIF_REMINDER_DISTRI(i,_colBlock,_colBlockRem);
        if (allocSpMatrixInternal(A->M,colBlock,colParts + i)){
            ERRPRINT("allocSpMatrixInternal of a A cols partition errd\n");
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
    //for each A col part. -> last copied nz value index
    //TODO also to get cols parts sizes to realloc overallocd AS,JA of A parts
    uint* colPartsLens = alloca(gridCols * sizeof(colPartsLens));
    memset(colPartsLens, 0, sizeof(*colPartsLens) * gridCols);
    //OFFSET BASED COPY OF A.COL_GROUPS -> O( A.NZ )
    for (uint r=0, j=0; r<A->M; j=A->IRP[++r]){
        //copy until the to the next  column group inside current row
        //continuing the A's nnz entries navigation (nz_idx=j+i)
        for (uint gc=0,gcStartCol=0,i=0;  gc<gridCols;  gc++,j+=i,i=0){
            colPart = colParts + gc;
            colPart->IRP[r] = colPartsLens[gc];
            gcStartCol += UNIF_REMINDER_DISTRI(gc,_colBlock,_colBlockRem);
            //@i=len current subpartition of row @r to compute
            for (uint c=A->JA[j+i]; c<gcStartCol && j+i<A->IRP[r+1]; c=A->JA[j + ++i]);
            memcpy(colPart->AS+colPartsLens[gc], A->AS+j, i*sizeof(*(A->AS)));
            memcpy(colPart->JA+colPartsLens[gc], A->JA+j, i*sizeof(*(A->JA)));
            
#ifdef ROWLENS
            colPart->RL[r] = i;
#endif
            colPartsLens[gc] += i;
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
 * sparsify accumulated vector @accV into sparse matrix row [partition] @accSparse
 * @accV has non zero values and indexed shifted back of @startColAcc columns
 */
static inline int sparsifyDenseVect(THREAD_AUX_VECT* accV,SPMATROW* accSparse,
  uint startColAcc){
    uint nnz = accV -> nnzIdxLast;
    accSparse -> len = nnz;
    //alloc row nnz element space
    if (!(accSparse->AS = malloc(nnz * sizeof(*(accSparse->AS))))){
        ERRPRINT("sparsifyDenseVect: accSparse->AS malloc error\n");
        goto _err;
    }
    if (!(accSparse->JA = malloc(nnz * sizeof(*(accSparse->JA))))){
        ERRPRINT("sparsifyDenseVect: accSparse->JA malloc error\n");
        goto _err;
    }

    sortuint(accV->nnzIdx,nnz); //sort nnz idx for ordered write

    ///SPARSIFY DENSE ACC.V INTO ROW
    for (uint i=0,j;    i<nnz;   i++){ 
        j = accV -> nnzIdx[i];        //shifted idx of a nnz of sp.Vect accumulator
        accSparse -> JA[i] = j + startColAcc;
        accSparse -> AS[i] = accV->v[j];
    }
    CONSISTENCY_CHECKS{ //TODO check if nnzIdxs sparsify is good as a full scan
        double* tmpNNZ = calloc(accV->vLen, sizeof(*tmpNNZ)); //TODO OVERFLOW POSSIBLE
        if (!tmpNNZ){ERRPRINT("sparsify check tmpNNZ malloc err\n");goto _err;}
        uint ii=0;
        for (uint i=0; i< accV->vLen; i++){
            if ( accV->v[i] )   tmpNNZ[ii++] = accV->v[i];
        }
        if (memcmp(tmpNNZ,accSparse->AS,nnz*sizeof(*tmpNNZ)))
        //if (doubleVectorsDiff(tmpNNZ,accSparse->AS,nnz))
            //{ERRPRINT("quick sparsify check err\n");free(tmpNNZ);goto _err;}
        if (ii != nnz){
            fprintf(stderr,"quick sparsify nnz num wrong!:%u<->%u\n",ii,nnz);
            free(tmpNNZ);goto _err;
        }
        free(tmpNNZ);
    }
    return EXIT_SUCCESS;
    _err:
    //if(accSparse->AS)  free(accSparse->AS); //TODO free anyway later
    //if(accSparse->JA)  free(accSparse->JA);
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
    
//////////////////////// COMPUTE CORE /////////////////////////////////////////
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
   
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint b,startRow,block,err; //omp for aux vars
    #pragma omp parallel for schedule(static) private(startRow,block)
    for (b=0;   b < conf->gridRows; b++){
        block      = UNIF_REMINDER_DISTRI(b,rowBlock,rowBlockRem);
        startRow   = UNIF_REMINDER_DISTRI_STARTIDX(b,rowBlock,rowBlockRem);
       
        DEBUG{
            fflush(NULL);
            printf("block %u\t%u:%u(%u)\n",b,startRow,startRow+block-1,block);
            fflush(NULL);
        }
        //row-by-row formulation in the given row block
        for (uint r=startRow;  r<startRow+block;  r++){
            //iterate over nz entry index c inside current row r
            for (uint c=A->IRP[r]; c<A->IRP[r+1]; c++) //row-by-row formul. accumulation
                scSparseRowMul(A->AS[c], B, A->JA[c], accVect+b);
            //trasform accumulated dense vector to CSR with using the aux idxs
            if ((err=sparsifyDenseVect(accVect+b,accRows + r,0))){
                fprintf(stderr,"sparsify failed ar row %u.\t aborting.\n",r);
                #pragma omp cancel for
            }
            _resetAccVect(accVect+b);   //rezero for the next A row
        }
    }
    if (err)                      goto _err;
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
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPMATROW* accRowsParts=NULL, *accRowPart;
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
     
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
    
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint tileID,t_i,t_j,err=0;    //for aux vars
    #pragma omp parallel for schedule(static) \
      private(accV,accRowPart,rowBlock,colBlock,startRow,startCol,\
      bPartLen,bPartID,bPartOffset,t_i,t_j)
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
        
        accV = accVectors + tileID; 
        if ((err=_allocAuxVect(accV,colBlock))){
            #pragma omp cancel for
        }
         
        DEBUG{
            fflush(NULL);
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

            accRowPart = accRowsParts + IDX2D(r,t_j,conf->gridCols);
            if ((err=sparsifyDenseVect(accV,accRowPart,startCol))){
                fprintf(stderr,"err sparsify at block %u,%u\n",t_i,t_j);
                #pragma omp cancel for
            }
            _resetAccVect(accV);
        }
    }
    if (err)                                        goto _err;
    if (mergeRowsPartitions(accRowsParts,AB,conf))  goto _err;
    goto _free;

    _err:
    if (AB) freeSpmat(AB);
    AB = NULL; 
    _free:
    if (offsets)     free(offsets);
    if (accRowsParts){
        for(uint i=0;i<subRowsN;i++)    freeSpRow(accRowsParts+i);
        free(accRowsParts);
    }
    _freeAccVectorsChecks(accVectors,gridSize);
    
    return AB;
        
}




spmat* spgemmGustavson2DBlocksAllocated(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm rowBlocks of A, (allcd) colBlocks of B\tM=%uxN=%u\n",A->M,B->N);
    spmat *AB = NULL, *colPartsB = NULL, *colPart;
    if (!(AB = allocSpMatrix(A->M,B->N)))   goto _err;
    //2D indexing aux vars
    uint gridSize=conf->gridRows*conf->gridCols, subRowsN=B->M*conf->gridCols;
    uint _rowBlock = AB->M/conf->gridRows, _rowBlockRem = AB->M%conf->gridRows;
    uint _colBlock = AB->N/conf->gridCols, _colBlockRem = AB->N%conf->gridCols;
    uint startRow,startCol,rowBlock,colBlock; //data division aux variables
    //aux vectors  
    THREAD_AUX_VECT *accVectors=NULL,*accV;
    SPMATROW* accRowsParts=NULL, *accRowPart;
    ////B cols  partition in CSRs
    if (!(colPartsB = colsPartitioningUnifRanges(B,conf->gridCols)))  goto _err;
    ///other AUX struct alloc
    //TODO aux vector of different sizes -> individual alloc later
    if (!(accVectors = calloc(gridSize, sizeof(*accVectors)))){
        ERRPRINT("accVectors calloc failed\n");
        goto _err;
    }
    if (!(accRowsParts = calloc(subRowsN, sizeof(*accRowsParts)))){
        ERRPRINT("accRowsParts calloc errd\n");
        goto _err;
    }
    
    AUDIT_INTERNAL_TIMES	Start=omp_get_wtime();
    uint tileID,t_i,t_j,err=0;    //for aux vars
    #pragma omp parallel for schedule(static) \
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
        if ((err=_allocAuxVect(accV,colBlock))){
            #pragma omp cancel for
        }
         
        DEBUG{
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

            accRowPart = accRowsParts + IDX2D(r,t_j,conf->gridCols);
            if ((err=sparsifyDenseVect(accV,accRowPart,startCol))){
                fprintf(stderr,"err sparsify at block %u,%u\n",t_i,t_j);
                #pragma omp cancel for
            }
            _resetAccVect(accV);
        }
    }
    if (err)                                        goto _err;
    if (mergeRowsPartitions(accRowsParts,AB,conf))  goto _err;
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

    return AB;
        
}


spmat* spgemmTemplate(spmat* A,spmat* B, CONFIG* conf){
    DEBUG printf("spgemm rowBlocks of A, (allcd) colBlocks of B\tM=%uxN=%u\n",A->M,B->N);
    spmat* AB = allocSpMatrix(A->M,B->N);
    if (!AB)    goto _err;
     

    _err:
    freeSpmat(AB);  AB = NULL; 
    _free:

    return AB;
}

spmat* sp3gemmGustavsonParallel(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    
    double end,start,elapsed,partial,flops;
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
    AUDIT_INTERNAL_TIMES    partial = omp_get_wtime() - Start;
    if (!(out = computeSpGEMM(RAC,P,conf)))     goto _free;
#ifdef DEBUG_TEST_CBLAS
    if(GEMMCheckCBLAS(RAC,P,out))               goto _free;
#endif
     
    end = omp_get_wtime();
    elapsed=end-start;
    flops = ( 2 * R->NZ * P->NZ * AC->NZ )/ ( elapsed );
    DEBUG 
      printf("sp3gemmGustavsonParallel of R:%ux%u AC:%ux%u P:%ux%u CSR sp.Mat",
        R->M,R->N,AC->M,AC->N,P->M,P->N);
    VERBOSE
      printf("elapsed %le - flops %le\tinternalTime: %le",elapsed,flops,end-Start+partial);
    

    _free:
    if (RAC)    freeSpmat(RAC);

    return out;

}
