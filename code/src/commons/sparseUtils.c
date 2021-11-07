#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"

int spmatDiff(spmat* A, spmat* B){
    if (A->NZ != B->NZ){
        ERRPRINT("NZ differ\n");
        return EXIT_FAILURE;
    }
    if (doubleVectorsDiff(A->AS,B->AS,A->NZ)){
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
#ifdef ROWLENS
    for (ushort i=0; i<spMatrix->M; i++)    printf("%u\t%u\n",i,spMatrix->RL[i]);
#endif
    printMatrix(denseMat,spMatrix->M,spMatrix->N,justNZMarkers);
    free(denseMat);
}

void print3SPGEMMCore(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    printf("@COARSENING AC: %ux%u ---> %ux%u\tconf grid: %ux%u,\tNNZ:%u-%u-%u\t",
      AC->M,AC->N, R->M,P->N, conf->gridRows,conf->gridCols, R->NZ,AC->NZ,P->NZ);
}
