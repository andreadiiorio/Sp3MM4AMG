/*
 * Copyright Andrea Di Iorio 2022
 * This file is part of Sp3MM_for_AlgebraicMultiGrid
 * Sp3MM_for_AlgebraicMultiGrid is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Sp3MM_for_AlgebraicMultiGrid is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Sp3MM_for_AlgebraicMultiGrid.  If not, see <http://www.gnu.org/licenses/>.
 */ 
#include <stdlib.h>
#include <stdio.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"
#include "SpMM_CBLAS.h"

#include <cblas.h>
////CBLAS - DENSE CHECKs
double* dgemmCBLAS(double* A,double* B, uint m, uint n, uint k){
    VERBOSE printf("computing A*B m:%u,n:%u,k:%u CBLAS gemm\n",m,n,k);
    double* out= malloc( m * n * sizeof(*out) );
    if (!out)  {ERRPRINT("dgemmCBLAS:\toracleOut malloc failed\n");return NULL;}
    CBLAS_LAYOUT layout=CblasRowMajor;
    CBLAS_TRANSPOSE notrans=CblasNoTrans;
    cblas_dgemm(layout,notrans,notrans, m, n, k, 1.0, A, k, B, n, 1.0, out, n);
    return out;
}
//wrap R,AC,P conversion to dense and compute RACP with CBLAS
double* sp3gemmToDenseCBLAS(spmat* R,spmat* AC,spmat* P){
    double *r=NULL, *ac=NULL, *p=NULL, *out=NULL, *rac=NULL;
    if (!(r=CSRToDense(R)) || !(ac=CSRToDense(AC))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: r, ac failed\n");
        goto _free;
    }
    if (!(rac = dgemmCBLAS(r,ac,R->M,AC->N,R->N)))  goto _free;
    free(r);    r   = NULL;
    free(ac);   ac  = NULL;
    if (!(p=CSRToDense(P))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: p failed\n");
        goto _free;
    }
    if (!(out = dgemmCBLAS(rac,p,R->M,P->N,AC->N))) goto _free;
    free(rac);  rac = NULL;
    free(p);    p   = NULL;
    
    _free:
    if (r)          free(r);
    if (ac)         free(ac);
    if (rac)        free(rac);
    if (p)          free(p);

    return out;
}
int sparseDenseMatrixCmp(spmat* spMat, double* denseMat){
    int ret = EXIT_FAILURE;
    double* mat = CSRToDense(spMat);
    if (!mat){
        ERRPRINT("sparseDenseMatrixCmp spMat to dense conversion failed\n");
        return ret;
    }
    ret = doubleVectorsDiff(mat,denseMat,spMat->M * spMat->N,NULL);
    _free:
    free(mat);
    return ret;
} 
int GEMMTripleCheckCBLAS(spmat* R,spmat* AC,spmat* P,spmat* RACP){
    int ret = EXIT_FAILURE;
    double* out=NULL;
    VERBOSE printf("checking parallel implementation using LAPACK.CBLAS\n");
    if (!(out = sp3gemmToDenseCBLAS(R,AC,P))) goto _free;
    ret = sparseDenseMatrixCmp(RACP,out);
    
    _free:
    if (out)        free(out);
    return ret;
}

int GEMMCheckCBLAS(spmat* A,spmat* B,spmat* AB){
    int ret = EXIT_FAILURE;
    double *a=NULL, *b=NULL, *outToCheck=NULL, *out=NULL;
    if (!(a=CSRToDense(A)) || !(b=CSRToDense(B))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: r, ac failed\n");
        goto _free;
    }
    if (!(out = dgemmCBLAS(a,b,A->M,B->N,A->N)))  goto _free;
    free(a);   a = NULL;
    free(b);   b = NULL;
    
    if (!(outToCheck = CSRToDense(AB) )){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: ab failed\n");
        goto _free;
    }
    ret = doubleVectorsDiff(outToCheck,out,AB->M * AB->N,NULL);
    _free:
    if (a)          free(a);
    if (b)          free(b);
    if (out)        free(out);
    if (outToCheck) free(outToCheck);

    return ret;
}
