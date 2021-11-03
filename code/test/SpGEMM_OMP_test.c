#include <stdlib.h>
#include <stdio.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"
#include "SpGEMM_OMP_test.h"

#ifdef CBLAS_TESTS
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
    ret = doubleVectorsDiff(mat,denseMat,spMat->M * spMat->N);
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
    ret = doubleVectorsDiff(outToCheck,out,AB->M * AB->N);
    _free:
    if (a)          free(a);
    if (b)          free(b);
    if (out)        free(out);
    if (outToCheck) free(outToCheck);

    return ret;
}
#endif

#ifdef TEST_MAIN
///MAIN WITH TESTS CHECKS
#include <string.h>
#include <omp.h>
#include "parser.h"
#include "SpGEMM.h"

////inline export here 
//spmat* allocSpMatrix(uint rows, uint cols);
//int allocSpMatrixInternal(uint rows, uint cols, spmat* mat);
//void freeSpmatInternal(spmat* mat);

#ifdef CBLAS_TESTS 
#define TESTTESTS "TESTTESTS"
#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},[AC_{i+1}," TESTTESTS "]" \
"all matriexes in MatrixMarket_sparse_matrix_COO\n" \
"giving AC_{i+1} the outputs of the 3SPGEMM will be matched with the given result as an oracle" \
"\ngiving" TESTTESTS "the tests function output over the given inputs will be used to check if match the given result AC_{i+1}\n" 

#else 

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},AC_{i+1}" \
"all matriexes in MatrixMarket_sparse_matrix_COO\n" \

#endif

//global vars   ->  audit
double Start,End,Elapsed,ElapsedInternal;

static CONFIG Conf = {
    .gridRows  = 8,
    .gridCols  = 8,
};

int main(int argc, char** argv){
    int ret=EXIT_FAILURE;
    if (init_urndfd())  return ret;
    if (argc < 4 )  {ERRPRINT(HELP); return ret;}
    
    double end,start,elapsed,flops;
    start = omp_get_wtime();

    spmat *R = NULL, *AC = NULL, *P = NULL, *outToCheck = NULL, *oracleOut=NULL;
    double* outCBLAS = NULL; //CBLAS TEST FUNCTION OUTPUT
    ////parse sparse matrixes 
    if (!( R = MMtoCSR(argv[1]))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        goto _free;
    }
    if (!( AC = MMtoCSR(argv[2]))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        goto _free;
    }
    if (!( P = MMtoCSR(argv[3]))){
        ERRPRINT("err during conversion MM -> CSR of P\n");
        goto _free;
    }
    if (argc > 4 ){  //get the result matrix to check computations
        if (!( oracleOut = MMtoCSR(argv[4]) )){
            ERRPRINT("err during conversion MM -> CSR AC_{i+1}\n");
            goto _free;
        }
    } 
    CONSISTENCY_CHECKS{     ///DIMENSION CHECKS...
        if (R->N != AC ->M){
            ERRPRINT("invalid sizes in R <-> AC\n");
            goto _free;
        }
        if (AC->N != P ->M){
            ERRPRINT("invalid sizes in AC <-> P\n");
            goto _free;
        }
        if (oracleOut){
            if (R->M != oracleOut->M){
                ERRPRINT("oracleOut invalid rows number\n");
                goto _free;
            }
            if (P->N != oracleOut->N){
                ERRPRINT("oracleOut invalid cols number\n");
                goto _free;
            }
        }
    }
    DEBUGPRINT{
        printf("sparse matrix: R_{i+1}\n"); printSparseMatrix(R,TRUE);
        printf("sparse matrix: AC_i\n");    printSparseMatrix(AC,TRUE);
        printf("sparse matrix: P_{i+1}\n"); printSparseMatrix(P,TRUE);
        if (argc > 4){ 
            printf("sparse matrix: AC_{i+1}\n");printSparseMatrix(oracleOut,TRUE);
        }
    }
#ifdef CBLAS_TESTS
    ////TEST THE TESTs 
    if ( argc == 6 && !strncmp(argv[5],TESTTESTS,strlen(TESTTESTS)) ){
        printf("testing the tests:\n matching dense CBLAS test with an oracle output\n");
        if (GEMMTripleCheckCBLAS(R,AC,P,oracleOut)){
            ERRPRINT("GEMMTripleCheckCBLAS matching with oracleOut failed\n");
            goto _free;
        }
        //TODO NEW TESTS CHECK HERE...
        goto _end;
    }
    //// PARALLEL COMPUTATIONs
    if (!oracleOut){
        if (!(outCBLAS = sp3gemmToDenseCBLAS(R,AC,P)))  goto _free;
    }
#endif 
    if (!getConfig(&Conf)){
        VERBOSE printf("configuration changed from env");
    }
    Conf.threadNum = (uint) omp_get_max_threads();
    SP3GEMM_INTERF sp3GEMMcompute=&sp3gemmGustavsonParallel;//TODO ITERATE OVER NEW POSSIBILITIES
    double times[AVG_TIMES_ITERATION],  timesInteral[AVG_TIMES_ITERATION];
    double elapsedStats[2],  elapsedInternalStats[2];
    //TODO int maxThreads = omp_get_max_threads();
    SPGEMM_INTERF spgemmFunc;   //TODO ITERATE OVER ALL SPGEMM COMPUTE FUNCTIONS
    end = omp_get_wtime();elapsed = end-start;
    VERBOSE printf("preparing time: %le\t",elapsed);
    print3SPGEMMCore(R,AC,P,&Conf);
    printf("stats samples size: %u\n",AVG_TIMES_ITERATION);
    uint f;
    for (f=0,spgemmFunc=SpgemmFuncs[f]; spgemmFunc; spgemmFunc=SpgemmFuncs[++f]){
        Conf.spgemmFunc = (void*) spgemmFunc; //spgemmFunc used twice in compute
        hprintf("@computing Sp3GEMM as pair of SpGEMM with func:\%u at:%p\t",
          f,spgemmFunc);
        for (uint i=0;  i< AVG_TIMES_ITERATION; i++){
            if (!(outToCheck = sp3GEMMcompute(R,AC,P,&Conf))){
                fprintf(stderr,"compute func number:%u failed...\n",f);
                goto _free;
            }
#ifdef CBLAS_TESTS
            if (!oracleOut){
                if (sparseDenseMatrixCmp(outToCheck,outCBLAS)){
                    fprintf(stderr,"compute func number:%u check with CBLAS fail\n",f);
                    goto _free;
                }
            } else {
#endif
            if (doubleVectorsDiff(outToCheck->AS,oracleOut->AS,oracleOut->NZ)){
                fprintf(stderr,"compute func number:%u check with oracle failed\n",f);
                goto _free;
            }
#ifdef CBLAS_TESTS
        }
#endif
            freeSpmat(outToCheck); outToCheck=NULL;
            times[i]        = Elapsed;
            timesInteral[i] = ElapsedInternal;
        }
        statsAvgVar(times,AVG_TIMES_ITERATION,elapsedStats);
        statsAvgVar(timesInteral,AVG_TIMES_ITERATION,elapsedInternalStats);
        printf("R:%ux%u AC:%ux%u P:%ux%u CSR sp.Mat ",
          R->M,R->N,AC->M,AC->N,P->M,P->N);
        printf("timeAvg:%le timeVar:%le\ttimeInternalAvg:%le timeInternalVar:%le \n",
            elapsedStats[0],elapsedStats[1],elapsedInternalStats[0],elapsedInternalStats[1]);

    }
    printf("\nall spgemmFuncs passed the test\n\n\n");


    DEBUGPRINT{
        printf("sparse matrix: AC_i\n");printSparseMatrix(outToCheck,TRUE);
    }
    _end:
    ret = EXIT_SUCCESS;
    _free:
    if (R)          freeSpmat(R);
    if (AC)         freeSpmat(AC);
    if (P)          freeSpmat(P);
    if (oracleOut)  freeSpmat(oracleOut);
    if (outCBLAS)   free(outCBLAS);
    if (outToCheck) freeSpmat(outToCheck);
    return ret;
}
#endif
