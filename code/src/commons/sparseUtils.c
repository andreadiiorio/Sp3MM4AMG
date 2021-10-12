#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#include "sparseMatrix.h"
#include "utils.h"
#include "macros.h"

///AUX
double* CSRToDense(spmat* sparseMat){
    double* denseMat;
    uint i,j,idxNZ;
    if (!(denseMat = calloc(sparseMat->M*sparseMat->N, sizeof(*denseMat)))){
        fprintf(stderr,"dense matrix alloc failed\n");
        return  NULL;
    }
    for (i=0;i<sparseMat->M;i++){
        for (idxNZ=sparseMat->IRP[i]; idxNZ<sparseMat->IRP[i+1]; ++idxNZ){
             j = sparseMat->JA[idxNZ];
             //converting sparse item into dense entry
             denseMat[IDX2D(i,j,sparseMat->N)] = sparseMat->AS[idxNZ]; 
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
