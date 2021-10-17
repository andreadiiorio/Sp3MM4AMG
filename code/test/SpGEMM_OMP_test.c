#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"
#include "SpGEMM_OMP_test.h"

double* dgemmCBLAS(double* A,double* B, uint m, uint n, uint k){
    VERBOSE printf("computing A*B m:%u,n:%u,k:%u CBLAS gemm\n",m,n,k);
    double* out= malloc( m * n * sizeof(*out) );
    if (!out)  {ERRPRINT("dgemmCBLAS:\toracleOut malloc failed\n");return NULL;}
    CBLAS_LAYOUT layout=CblasRowMajor;
    CBLAS_TRANSPOSE notrans=CblasNoTrans;
    cblas_dgemm(layout,notrans,notrans, m, n, k, 1.0, A, k, B, n, 1.0, out, n);
    return out;
}

int GEMMTripleCheckCBLAS(spmat* R,spmat* AC,spmat* P,spmat* RACP){
    int ret = EXIT_FAILURE;
    double *r=NULL, *ac=NULL, *p=NULL, *outToCheck=NULL, *out=NULL, *rac=NULL;
    if (!(r=CSRToDense(R)) || !(ac=CSRToDense(AC))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: r, ac failed\n");
        goto _free;
    }
    VERBOSE printf("checking parallel implementation using LAPACK.CBLAS\n");

    if (!(rac = dgemmCBLAS(r,ac,R->M,AC->N,R->N)))  goto _free;
    free(ac);   ac  = NULL;
    free(r);    r   = NULL;
    if (!(p=CSRToDense(P))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: p failed\n");
        goto _free;
    }
    if (!(out = dgemmCBLAS(rac,p,R->M,P->N,AC->N))) goto _free;
    free(rac);  rac = NULL;
    free(p);    p   = NULL;
    if (!(outToCheck=CSRToDense(RACP))){
        ERRPRINT("serialDenseGEMMTest aux dense matrix alloc: racp failed\n");
        goto _free;
    }

    ret = doubleVectorsDiff(outToCheck,out,RACP->M * RACP->N);
    
    _free:
    if (r)          free(r);
    if (ac)         free(ac);
    if (p)          free(p);
    if (out)        free(out);
    if (outToCheck) free(outToCheck);
    if (rac)        free(rac);

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
    ret = doubleVectorsDiff(outToCheck,out,AB->M * AB->N);
    _free:
    if (a)          free(a);
    if (b)          free(b);
    if (out)        free(out);
    if (outToCheck) free(outToCheck);

    return ret;
}

#ifdef TEST_MAIN
///MAIN WITH TESTS CHECKS
#include <omp.h>
#include "parser.h"
#include "SpGEMM.h"

extern void freeSpRow(SPMATROW* r);
extern void freeSpmat(spmat* mat);
COMPUTEFUNC spgemmRowsBasic;

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},[AC_{i+1} -> test the test" \
    "in MatrixMarket_sparse_matrix_COO"

CONFIG Conf = {
    .gridRows = 8,
    .threadNum = 8
};

int main(int argc, char** argv){
    int ret=EXIT_FAILURE;
    if (init_urndfd())  return ret;
    if (argc < 4 )  {ERRPRINT(HELP); return ret;}
    
    spmat *R = NULL, *AC = NULL, *P = NULL, *out = NULL;
    ////parse sparse matrixes 
    if (!( R = MMtoCSR(argv[1]))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        goto _free;
    }
    ////parse sparse matrixes 
    if (!( AC = MMtoCSR(argv[2]))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        goto _free;
    }
    ////parse sparse matrixes 
    if (!( P = MMtoCSR(argv[3]))){
        ERRPRINT("err during conversion MM -> CSR of P\n");
        goto _free;
    }
    
    CONSISTENCY_CHECKS{
        if (R->N != AC ->M){
            ERRPRINT("invalid sizes in R <-> AC\n");
            goto _free;
        }
        if (AC->N != P ->M){
            ERRPRINT("invalid sizes in AC <-> P\n");
            goto _free;
        }
    }
    DEBUG {
        printf("sparse matrix: R_i+1\n"); printSparseMatrix(R,TRUE);
        printf("sparse matrix: AC_i\n");printSparseMatrix(AC,TRUE);
        printf("sparse matrix: P_i+1\n") ;printSparseMatrix(P,TRUE);
    }
    
    COMPUTEFUNC_INTERF computeFunc=&spgemmRowsBasic; //TODO ITERATE OVER ALL POSSIBILITIES
    if (argc > 4 ){  //test the test with given 3multiplication result
        if (!( out = MMtoCSR(argv[4]) )){
            ERRPRINT("err during conversion MM -> CSR AC_{i+1}\n");
            goto _free;
        }
        VERBOSE printf("testing the dense CBLAS test with an oracle output\n");
    } else {    //// PARALLEL COMPUTATIONs
        //TODO int maxThreads = omp_get_max_threads();
        if (!(out = computeFunc(R,AC,P,&Conf))){
            ERRPRINT("compute function selected failed...\n");
            goto _free;
        }
    }

    //// SERIAL - LAPACK.CBLAS COMPUTATION
    if (GEMMTripleCheckCBLAS(R,AC,P,out)){
        ERRPRINT("LAPACK.CBLAS SERIAL, DENSE GEMM TEST FAILED!!\n");
        goto _free;
    }
    DEBUG{
        printf("sparse matrix: AC_i\n");printSparseMatrix(out,TRUE);
    }
    ret = EXIT_SUCCESS;
    _free:
    if (R)          freeSpmat(R);
    if (AC)         freeSpmat(AC);
    if (P)          freeSpmat(P);
    if (out)        freeSpmat(out);
    return ret;
}
#endif
