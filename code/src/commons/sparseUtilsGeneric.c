#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"

///////////////   no offset --> SINGLE implementation functions
#ifndef SPARSE_UTILS_C
#define SPARSE_UTILS_C
void checkOverallocRowPartsPercent(ulong* forecastedSizes,spmat* AB,
  idx_t gridCols,idx_t* bColOffsets){
	idx_t* abColOffsets = colsOffsetsPartitioningUnifRanges_0(AB,gridCols);
	assert(abColOffsets);	//partitioning error
    for (idx_t i=0,rLen,forecast,partId=0; i<AB->M*gridCols; i++,partId++){
        forecast = forecastedSizes[i];
		rLen = abColOffsets[ partId+1 ] -abColOffsets[ partId ];
		DEBUGCHECKS	assert(forecast >= rLen);
	}
    idx_t extraMatrix = forecastedSizes[AB->M] - AB->NZ;
    printf("extra forecastedSize of the matrix: \t%lu\t = %lf %% \n",
      extraMatrix, 100*extraMatrix /(double) forecastedSizes[AB->M]);
	
}

void checkOverallocPercent(ulong* forecastedSizes,spmat* AB){
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
            	assert(forecastedSize >= rSize );
            }
        }
        DEBUGPRINT
            printf("extra forecastedSize of row: %lu\t=\t%lf %% \n",
              r,100*(forecastedSize-rSize) / (double) forecastedSize);
    }
    idx_t extraMatrix = forecastedSizes[AB->M] - AB->NZ;
    printf("extra forecastedSize of the matrix: \t%lu\t = %lf %% \n",
      extraMatrix, 100*extraMatrix /(double) forecastedSizes[AB->M]);
}
int spmatDiff(spmat* A, spmat* B){
    if (A->NZ != B->NZ){
        ERRPRINT("NZ differ\n");
        return EXIT_FAILURE;
    }
    if (doubleVectorsDiff(A->AS,B->AS,A->NZ,NULL)){
        ERRPRINT("AS DIFFER\n");
        return EXIT_FAILURE;
    }
    if (memcmp(A->JA,B->JA,A->NZ)){
        ERRPRINT("JA differ\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

double* CSRToDense(spmat* sparseMat){
    double* denseMat;
    ulong i,j,idxNZ,denseSize;
    if (__builtin_umull_overflow(sparseMat->M,sparseMat->N,&denseSize)){
        ERRPRINT("overflow in dense allocation\n");
        return NULL;
    }
    if (!(denseMat = calloc(denseSize, sizeof(*denseMat)))){
        ERRPRINT("dense matrix alloc failed\n");
        return  NULL;
    }
    for (i=0;i<sparseMat->M;i++){
        for (idxNZ=sparseMat->IRP[i]; idxNZ<sparseMat->IRP[i+1]; ++idxNZ){
             j = sparseMat->JA[idxNZ];
             //converting sparse item into dense entry
             denseMat[(ulong) IDX2D(i,j,sparseMat->N)] = sparseMat->AS[idxNZ]; 
        }
    }
    return denseMat;
}
void printSparseMatrix(spmat* spMatrix,char justNZMarkers){
    double* denseMat = CSRToDense(spMatrix);
    if (!denseMat)  return;
    printMatrix(denseMat,spMatrix->M,spMatrix->N,justNZMarkers);
    free(denseMat);
}

void print3SPMMCore(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    printf("@COARSENING AC: %lux%lu ---> %lux%lu\t"
      "conf grid: %ux%u,\tNNZ:%lu-%lu-%lu\t AVG_TIMES_ITERATION:%u\n",
      AC->M,AC->N, R->M,P->N, conf->gridRows,conf->gridCols,
      R->NZ,AC->NZ,P->NZ,AVG_TIMES_ITERATION);
}


static inline int _colsPartitioningUnifRanges_init(spmat* A,uint gridCols,
  spmat** colParts,idx_t** colPartsLens){

    spmat* colPart;
    ulong _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols, *tmpJA;
    ///alloc/init partitions structures
    if (!(*colParts = calloc(gridCols, sizeof(**colParts)))){
        ERRPRINT("colsPartitioningUnifRanges\tcolumns partitions of A calloc fail\n");
        return EXIT_FAILURE;
    }
    for (ulong i=0,colBlock; i<gridCols; i++){
        colBlock = UNIF_REMINDER_DISTRI(i,_colBlock,_colBlockRem);
        colPart  = *colParts + i;
        if (allocSpMatrixInternal(A->M,colBlock,colPart)){
            ERRPRINT("colsPartitioningUnifRanges\tallocSpMatrixInternal partition err\n");
        	return EXIT_FAILURE;
        }
        //TODO TODO overalloc A cols partitions NZ arrays, then realloc
        if (!(colPart->AS = malloc(A->NZ * sizeof(*A->AS)))){
            ERRPRINT("colPart of A overalloc of AS errd\n");
        	return EXIT_FAILURE;
        }
        if (!(colPart->JA = malloc(A->NZ * sizeof(*A->JA)))){
            ERRPRINT("colPart of A overalloc of JA errd\n");
        	return EXIT_FAILURE;
        }
    }
    //for each A col partition -> last copied nz index = nnz copied ammount
    if (! (*colPartsLens = calloc(gridCols, sizeof(**colPartsLens))) ) {
        ERRPRINT("colsPartitioningUnifRanges: colPartsLens calloc errd\n");
        return EXIT_FAILURE;
    }
	return EXIT_SUCCESS;
}

static inline int _colsPartitioningUnifRanges_finalRealloc(spmat* A,uint gridCols,
  spmat* colParts,idx_t* colPartsLens){

	spmat* colPart;
	double* tmpAS; idx_t* tmpJA;
    //realloc overallcd A parts NZ arrays (TODO -> downsizing -> nofails?)
    for (ulong i=0,partLen; i<gridCols; i++){
        colPart = colParts + i;
        partLen = colPartsLens[i];
        if (!(tmpAS = realloc(colPart->AS,partLen*sizeof(*(colPart->AS))))){
            ERRPRINT("realloc overallocated cols partition AS array\n");
            return EXIT_FAILURE;
        }
        colPart->AS = tmpAS;
        if (!(tmpJA = realloc(colPart->JA,partLen*sizeof(*(colPart->JA))))){
            ERRPRINT("realloc overallocated cols partition JA array\n");
            return EXIT_FAILURE;
        }
        colPart->JA         = tmpJA;
        colPart->NZ         = partLen;
        colPart->IRP[A->M]  = partLen;
    }
	return EXIT_SUCCESS;
}
#endif

 
#ifndef OFF_F
    #error generic implementation requires OFF_F defined
#endif
////////////////////////  CSR SPECIFIC -- TODO RENAME //////////////////
///SPARSE MATRIX PARTITIONING
idx_t* CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(spmat* A,uint gridCols){
    ulong subRowsN = A->M * gridCols;
    ulong _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols;
    ulong* offsets = malloc( (subRowsN+1) * sizeof(*offsets) );
    if (!offsets)  {
        ERRPRINT("colsOffsetsPartitioningUnifRanges:\toffsets malloc errd\n");
        return NULL;
    }
    ///OFFSETS COMPUTE FOR COL GROUPS -> O( A.NZ )
    for (ulong r=0, j=0;     r<A->M;     j=A->IRP[++r]-OFF_F){
        offsets[ IDX2D(r,0,gridCols) ] = j;  //row's first gc start is costrained
        //navigate column groups inside current row
        for (ulong gc=1,gcStartCol;  gc<gridCols;  gc++){
            gcStartCol = UNIF_REMINDER_DISTRI_STARTIDX(gc,_colBlock,_colBlockRem);
            //goto GroupCols start entry,keeping A's nnz entries navigation (idx j)
            //for (ulong c=A->JA[j]-OFF_F; c<gcStartCol && j < A->IRP[r+1]-OFF_F; c=A->JA[++j]-OFF_F);
            while ( j < A->IRP[r+1]-OFF_F &&  A->JA[j]-OFF_F < gcStartCol )  j++;
            offsets[ IDX2D(r,gc,gridCols) ] = j;  //row's gc group startIdx
        }
    }
    offsets[subRowsN] = A->NZ;  //last row's partition end costrained
    return offsets;
}

spmat* CAT(colsPartitioningUnifRangesOffsetsAux_,OFF_F)(spmat* A,uint gridCols,idx_t** colPartsOffsets){
    spmat *colParts, *colPart;
    ulong _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols, *colPartsLens=NULL, *tmpJA;
    double* tmpAS;
    ///alloc/init partitions structures
    idx_t* colOffsets  = NULL;
    if (!(colOffsets = CAT(colsOffsetsPartitioningUnifRanges_,OFF_F)(A,gridCols))) goto _err;
    if (_colsPartitioningUnifRanges_init(A,gridCols,&colParts,&colPartsLens))			 goto _err;
    //OFFSET BASED COPY OF A.COL_GROUPS -> O( A.NZ )
    for (idx_t r=0,gcId=0;     r<A->M;    r++){
        for (idx_t gc=0,gcStartIdx=0,gLen=0;  gc<gridCols; gc++,gcId++){
			//gcId			= IDX2D(r,gc,gridCols); //seqent. scan of parts
			colPart			= colParts + gc;
			gcStartIdx 		= colOffsets[ gcId   ];
			gLen			= colOffsets[ gcId+1 ] - gcStartIdx;
            colPart->IRP[r] = colPartsLens[gc];	//new line for the col partition
			//actual copy of nnz entries to colPartitions
            memcpy(colPart->AS+colPart->IRP[r], A->AS+gcStartIdx, gLen*sizeof(*A->AS));
            memcpy(colPart->JA+colPart->IRP[r], A->JA+gcStartIdx, gLen*sizeof(*A->JA));
            colPartsLens[gc] += gLen;
			#ifdef ROWLENS
            colPart->RL[r] = i;
			#endif
		}
	}

    //realloc overallcd A parts NZ arrays
    if(_colsPartitioningUnifRanges_finalRealloc(A,gridCols,colParts,colPartsLens)) 		 goto  _err;

    free(colPartsLens);
	if (colPartsOffsets)	*colPartsOffsets = colOffsets;	//save for the caller
	else					free(colPartsOffsets);

    return colParts;
    _err:
    if(*colPartsOffsets)				free(*colPartsOffsets);
    for (ulong i=0; i<gridCols; i++)   	freeSpmatInternal(colParts+i);
    if(colParts)        				free(colParts);
    if(colPartsLens)    				free(colPartsLens);
    return NULL;
}
spmat* CAT(colsPartitioningUnifRanges_,OFF_F)(spmat* A,uint gridCols){
    spmat *colParts, *colPart;
    ulong _colBlock = A->N/gridCols, _colBlockRem = A->N%gridCols, *colPartsLens=NULL, *tmpJA;
    double* tmpAS;
    ///alloc/init partitions structures
    if (_colsPartitioningUnifRanges_init(A,gridCols,&colParts,&colPartsLens))	goto _err;
    /* TODO
     * Parallelize: 2for collapse OMP, gcEndCol -> startIdxize, ...
     * oppure wrappare cio in static inline 
     */
    for (ulong r=0, j=0;     r<A->M;     j=A->IRP[++r]-OFF_F){
        //navigate column groups inside current row
        for (ulong gc=0,gcEndCol=0,i;  gc<gridCols ;  gc++,j+=i){
            i = 0;  //@i=len current subpartition of row @r to copy
            colPart = colParts + gc;
			//NB: not shifting IRP because is handled as internal implementation component
 			//But JA idx memcopied -> kept as they were originally, handled with shift in functions
            colPart->IRP[r] = colPartsLens[gc];	
            gcEndCol += UNIF_REMINDER_DISTRI(gc,_colBlock,_colBlockRem);
            //goto next GroupCols,keeping A's nnz entries navigation ( index j+i )
            //for (ulong c=A->JA[j+i]-OFF_F; c<gcEndCol && j+i  < A->IRP[r+1]-OFF_F; c=A->JA[j+ ++i]-OFF_F);
            while ( j+i < A->IRP[r+1]-OFF_F && A->JA[j+i]-OFF_F < gcEndCol ) i++;
            memcpy(colPart->AS+colPart->IRP[r], A->AS+j, i*sizeof(*A->AS));
            memcpy(colPart->JA+colPart->IRP[r], A->JA+j, i*sizeof(*A->JA));
            
            colPartsLens[gc] += i;
			#ifdef ROWLENS
            colPart->RL[r] = i;
			#endif
        }
    }
    //realloc overallcd A parts NZ arrays
    if(_colsPartitioningUnifRanges_finalRealloc(A,gridCols,colParts,colPartsLens)) goto  _err;
    free(colPartsLens);
    return colParts;
    _err:
    for (ulong i=0; i<gridCols; i++)   freeSpmatInternal(colParts+i);
    if(colParts)        free(colParts);
    if(colPartsLens)    free(colPartsLens);
    return NULL;
}



#ifdef SPARSEUTILS_MAIN_TEST	///unit test embbeded

///inline export here 
//SPMV_CHUNKS_DISTR spmvChunksFair; 
spmat* allocSpMatrix(ulong rows, ulong cols);
int allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
void freeSpmatInternal(spmat* mat);
void freeSpmat(spmat* mat);

////INTERNAL TEST FUNCTIONS
//test that each row's partition from colsOffsetsPartitioningUnifRanges is in the correct index range
#include <alloca.h>
int testColsOffsetsPartitioningUnifRanges(spmat* mat,ulong gridCols,ulong* partsOffs){
    ulong _colBlock = mat->N/gridCols, _colBlockRem = mat->N%gridCols;
    ulong j=0;    //CSR scanning nnz idx
    ulong* colPartsPopulations = alloca(gridCols * sizeof(*colPartsPopulations));
    memset(colPartsPopulations,0,gridCols * sizeof(*colPartsPopulations));
    for (ulong r=0,pId=0; r<mat->M; r++){
        for (ulong gc=0,pStartIdx,pEndIdx; gc<gridCols; gc++,pId++){
            pStartIdx = UNIF_REMINDER_DISTRI_STARTIDX(gc,_colBlock,_colBlockRem);
            pEndIdx   = UNIF_REMINDER_DISTRI_STARTIDX(gc+1,_colBlock,_colBlockRem)-1; 
            //pId=IDX2D(r,gc,gridCols);
            for (ulong idx=partsOffs[pId],c; idx<partsOffs[pId+1]; idx++,j++){
                c = mat->JA[idx];
                assert( j == idx ); //consecutive index in partitioning
                assert( pStartIdx <= c && c <= pEndIdx );               //colRange
                assert( mat->IRP[r] <= idx && idx <= mat->IRP[r+1] );   //rowRange
            }
            colPartsPopulations[gc] += partsOffs[pId+1] - partsOffs[pId]; 
        }
    }
    assert( j == mat->NZ );
    ulong s=0;
    for (ulong gc=0,partSize; gc < gridCols; gc++,s+=partSize){
        partSize = colPartsPopulations[gc];
        double partShare=partSize/(double)mat->NZ,partsAvg=1/(double)gridCols;
        double partShareAvgDiff = partShare - partsAvg;
        printf("colPartition %lu has:\t%lu = %lf of NNZ\t\t .. %lf\tAVG diff\n",
          gc,partSize,partShare,partShareAvgDiff);
    }
    assert( s == mat->NZ ); //TODO DUPLICATED
    return EXIT_SUCCESS;
}

CONFIG Conf = {
    .gridRows = 8,
    .gridCols = 8,
};

#include "parser.h"
int main(int argc, char** argv){
    int out=EXIT_FAILURE;
    if (init_urndfd())  return out;
    if (argc < 2 )  {ERRPRINT("COO MATRIX FOR TEST"); return out;}
    ////parse sparse matrix and dense vector
    spmat* mat;
    char* trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[1],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[1];
    if (!(mat = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR\n");
        return out;
    }
    ////partitioning test
    ulong* colsPartitions = colsOffsetsPartitioningUnifRanges_0(mat,Conf.gridCols);
    if (!colsPartitions)    goto _free;
    if (testColsOffsetsPartitioningUnifRanges(mat,Conf.gridCols,colsPartitions))  goto _free;

    out=EXIT_SUCCESS;
    printf("testColsOffsetsPartitioningUnifRanges passed with "
           "mat: %lux%lu-%luNNZ\tgrid: %dx%d\n",
            mat->M,mat->N,mat->NZ,Conf.gridRows,Conf.gridCols);
    _free:
    if (colsPartitions) free(colsPartitions);

    return out;
}
#endif //SPARSEUTILS_MAIN_TEST