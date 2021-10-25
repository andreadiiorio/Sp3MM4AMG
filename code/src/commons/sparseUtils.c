#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"

///AUX
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
    printf("@COARSENING AC: %ux%u ---> %ux%u\tconf grid: %ux%u,\tNNZ:%u-%u-%u",
      AC->M,AC->N, R->M,P->N, conf->gridRows,conf->gridCols, R->NZ,AC->NZ,P->NZ);
}
