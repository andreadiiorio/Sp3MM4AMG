//sparse matrix def & aux
#ifndef SPARSEMATRIX
#define SPARSEMATRIX 

#include <stdlib.h>
#include <string.h>

#include "macros.h"
#include "config.h"

typedef struct {
    idx_t NZ,M,N;
    double *AS; 
    idx_t* JA;
    //CSR SPECIFIC
    idx_t* IRP;
	#ifdef ROWLENS
    idx_t* RL;   //row lengths
	#endif
    //CUDA SPECIFIC
    idx_t MAX_ROW_NZ;

} spmat; //describe a sparse matrix

//smart index keeping in a dense map
typedef struct{
	idx_t	len;				//num of nnz idx accumulated
	/* nnz index presence packing, implict space enough for all possible indexes*/
	nnz_idxs_flags_t idxsMap;
	uint idxsMapN;	//either num of limbs or len of char flag array
} SPVECT_IDX_DENSE_MAP;
int initSpVectIdxDenseAcc(idx_t idxMax, SPVECT_IDX_DENSE_MAP* );  
inline void _resetIdxMap(SPVECT_IDX_DENSE_MAP* acc){
	acc->len = 0;
	memset(acc->idxsMap,0,sizeof(*acc->idxsMap)*acc->idxsMapN);
}
inline void _freeIdxMap(SPVECT_IDX_DENSE_MAP* acc){
	free(acc->idxsMap);
	free(acc);
}
//aux struct for sparse vector-scalar product accumualtion
typedef struct{ 
    double*  v;          			//aux accumulating dense vector (sparse)
    idx_t*   nnzIdx;     			//v nnz value's indexes 		(contiguos)
    idx_t    vLen;       			//size of the aux dense vector //TODO USELESS?
    SPVECT_IDX_DENSE_MAP nnzIdxMap;	//
} ACC_DENSE;
int allocAccDense(ACC_DENSE* v, ulong size);

/*  
 **** NNZ IDXS PRESENCE FLAGS ACCESS INTERFACE:  ***
 *    sp_vect_idx_set(idx,SPVECT_IDX_DENSE_MAP)
 *    	-> return 0 if idx isn't already setted and in case set it
 */
static inline int spVect_idx_in(idx_t idx, SPVECT_IDX_DENSE_MAP* idxsMapAcc){
	#if SPVECT_IDX_BITWISE == TRUE
	uint limbID 	= idx / (sizeof(*idxsMapAcc->idxsMap) * 8); //idx's limb id
	uint limbIdxID	= idx % (sizeof(*idxsMapAcc->idxsMap) * 8); //idx's pos in limb
	limb_t idxPos   = ((limb_t) 1) << limbIdxID;
	if (!( idxsMapAcc->idxsMap[limbID] & idxPos) ){
		idxsMapAcc->idxsMap[limbID] |= idxPos;
		idxsMapAcc->len++;
		return 0;
	}
	#else	
	if (!( nnzIdxsFlags[idx] )){
		idxsMapAcc->idxsMap[idx] = 1;
		idxsMapAcc->len++;
		return 0;
	}
	#endif //SPVECT_IDX_BITWISE == TRUE
	return 1;
}	
inline void _resetAccVect(ACC_DENSE* acc){
    memset(acc->v,		 0,	acc->vLen * sizeof(*(acc->v)));
    memset(acc->nnzIdx,	 0,	acc->vLen * sizeof(*(acc->nnzIdx)));
    _resetIdxMap(&acc->nnzIdxMap);
}
////Sparse vector accumulator -- corresponding to a matrix portion
typedef struct{
    //idx_t    r;     //row index in the corresponding matrix
    //idx_t    c;     //col index in the corresponding matrix
    idx_t   len;   //rowLen
    double* AS;    //row nnz    values
    idx_t*  JA;    //row nnz    colIndexes
} SPACC; 


/*
 * ARRAY BISECTION - RECURSIVE VERSION
 * TODO ASSERT LEN>0 ommitted
 */
inline int BISECT_ARRAY(idx_t target, idx_t* arr, idx_t len){
    //if (len == 0)              return FALSE;
    if (len <= 1)              return *arr == target; 
    idx_t middleIdx = (len-1) / 2;  //len=5-->2, len=4-->1
    idx_t middle    = arr[ middleIdx ];
    if      (target == middle)  return TRUE;
    else if (target <  middle)  return BISECT_ARRAY(target,arr,middleIdx); 
    else    return BISECT_ARRAY(target,arr+middleIdx+1,middleIdx + (len-1)%2);
}

/*
 * return !0 if col @j idx is in row @i of sparse mat @smat
 * bisection used --> O(log_2(ROWLENGHT))
 */
inline int IS_NNZ(spmat* smat,idx_t i,idx_t j){
    idx_t rStart = smat->IRP[i];
    idx_t rLen   = smat->IRP[i+1] - rStart;
    if (!rLen)  return FALSE;
    return BISECT_ARRAY(j,smat->JA + rStart,rLen);
}
inline int IS_NNZ_linear(spmat* smat,idx_t i,idx_t j){    //linear -> O(ROWLENGHT)
    int out = 0;
    for (idx_t x=smat->IRP[i]; x<smat->IRP[i+1] && !out; x++){
        out = (j == smat->JA[x]); 
    } 
    return out;
}
////aux functions
//free sparse matrix
inline void freeSpmatInternal(spmat* mat){
    if(mat->AS)    free(mat->AS);  
    if(mat->JA)    free(mat->JA);  
    if(mat->IRP)   free(mat->IRP);  
#ifdef ROWLENS
    if(mat->RL)    free(mat->RL);
#endif 
}
inline void freeSpmat(spmat* mat){
    freeSpmatInternal(mat);
    free(mat);
}

//free max aux structs not NULL pointed
inline void freeSpAcc(SPACC* r){ 
    if(r->AS)   free(r->AS);
    if(r->JA)   free(r->JA);
}
////alloc&init functions
//alloc&init internal structures only dependent of dimensions @rows,@cols
inline int allocSpMatrixInternal(idx_t rows, idx_t cols, spmat* mat){
    mat -> M = rows;
    mat -> N = cols;
    if (!(mat->IRP=calloc(mat->M+1,sizeof(*(mat->IRP))))){ //calloc only for 0th
        ERRPRINT("IRP calloc err\n");
        freeSpmatInternal(mat);
        return EXIT_FAILURE;
    }
#ifdef ROWLENS
    if (!(mat->RL = malloc(mat->M*sizeof(*(mat->RL))))){
        ERRPRINT("RL calloc err\n");
        freeSpmatInternal(mat);
        return EXIT_FAILURE;
    }
#endif
    return EXIT_SUCCESS;
}

//alloc a sparse matrix of @rows rows and @cols cols 
inline spmat* allocSpMatrix(idx_t rows, idx_t cols){

    spmat* mat;
    if (!(mat = calloc(1,sizeof(*mat)))) { 
        ERRPRINT("mat  calloc failed\n");
        return NULL;
    }
    if (allocSpMatrixInternal(rows,cols,mat)){
        free(mat);
        return NULL;
    }
    return mat;
}

#endif
