#include <stdlib.h>
#include <stdio.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "utils.h"
#include "SpGEMM_test.h"

///MAIN WITH TESTS CHECKS
#include <string.h>
#include <omp.h>
#include "parser.h"
#include "SpGEMM.h"
#include "ompChunksDivide.h"
#include "ompGetICV.h"  //ICV - RUNTIME information audit auxs

////inline funcs
spmat*  allocSpMatrix(ulong rows, ulong cols);
int     allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
spmat*  initSpMatrixSpGEMM(spmat* A, spmat* B);
void    freeSpmatInternal(spmat* mat);
void    freeSpmat(spmat* mat);

CHUNKS_DISTR    chunksFair,chunksFairFolded,chunksNOOP;


//global vars   ->  audit
CHUNKS_DISTR_INTERF chunkDistrbFunc=&chunksFairFolded;
//double Start,End,Elapsed,ElapsedInternal;
static CONFIG Conf = {
    .gridRows  = 8,
    .gridCols  = 8,
};

/*
 * wrap result check and stats gather of Sp3GEMM implementation func at (@sp3gemm,[@spgemm])
 * result checked with @oracleOut or with CBLAS dense-serial implementation ifdef CBLAS_TESTS
 * @spgemm forwarded to @sp3gemm, stats printed on stdout; EXIT_FAILURE returned if any error
 */
static inline int testSp3GEMMImpl(SP3GEMM_INTERF sp3gemm,SPGEMM_INTERF spgemm,spmat* oracleOut,
  spmat* R,spmat* AC,spmat* P){
    int out = EXIT_FAILURE;
    spmat* outToCheck=NULL;
    //elapsed stats aux vars
    double times[AVG_TIMES_ITERATION],  timesInteral[AVG_TIMES_ITERATION];
    double deltaTStats[2],  deltaTInternalStats[2],internal_over_full,start,end;
    for (uint i=0;  i<AVG_TIMES_ITERATION; i++){
        start = omp_get_wtime();
        if (!(outToCheck = sp3gemm(R,AC,P,&Conf,spgemm))){
            ERRPRINTS("compute sp3gemm at:%p with spgemm at:%p failed...\n",sp3gemm,spgemm);
            return EXIT_FAILURE;
        }
        end = omp_get_wtime();

        #ifdef CBLAS_TESTS
        if (GEMMTripleCheckCBLAS(R,AC,P,outToCheck)){
             ERRPRINTS("compute sp3gemm at:%p with spgemm at:%p "
               "diff with CBLAS implementation...\n",sp3gemm,spgemm);
             goto _free;
        }
        #else
        if (spmatDiff(outToCheck,oracleOut))    goto _free;
        #endif

        freeSpmat(outToCheck); outToCheck = NULL;
        times[i]        = end - start;
        timesInteral[i] = ElapsedInternal;
        ElapsedInternal = Elapsed = 0;
    }
    statsAvgVar(times,AVG_TIMES_ITERATION,deltaTStats);
    statsAvgVar(timesInteral,AVG_TIMES_ITERATION,deltaTInternalStats);
    internal_over_full = deltaTInternalStats[0] / deltaTStats[0];
    //printf("R:%lux%lu AC:%lux%lu P:%lux%lu ",R->M,R->N,AC->M,AC->N,P->M,P->N);
    printf("timeAvg:%le timeVar:%le\ttimeInternalAvg:%le (%lf%% tot) timeInternalVar:%le \n",
     deltaTStats[0],deltaTStats[1],deltaTInternalStats[0],internal_over_full*100,deltaTInternalStats[1]);

    
    out = EXIT_SUCCESS;
    _free:
    if(outToCheck)  freeSpmat(outToCheck);
    return out;
}

#define TESTTESTS "TESTTESTS"
#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},[AC_{i+1}," TESTTESTS "(requires -DCBLAS_TEST)]"\
"all matriexes in MatrixMarket_sparse_matrix_COO[.compressed]\n" \
"giving AC_{i+1} the outputs of the 3SPGEMM will be matched with the given result as an oracle" \
"\ngiving" TESTTESTS "the tests function output over the given inputs will be used to check if match the given result AC_{i+1}\n" 


int main(int argc, char** argv){
    int ret=EXIT_FAILURE;
    if (init_urndfd())  return ret;
    if (argc < 4 )  {ERRPRINT(HELP); return ret;}
    
    double end,start,elapsed,flops;
    start = omp_get_wtime();

    spmat *R = NULL, *AC = NULL, *P = NULL, *outToCheck = NULL, *oracleOut=NULL;
    ////parse sparse matrixes 
    char* trgtMatrix;
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[1],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[1];
    if (!( R = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR of R\n");
        goto _free;
    }
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[2],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[2];
    if (!( AC = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR of AC\n");
        goto _free;
    }
    trgtMatrix = TMP_EXTRACTED_MARTIX;
    if (extractInTmpFS(argv[3],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[3];
    if (!( P = MMtoCSR(trgtMatrix))){
        ERRPRINT("err during conversion MM -> CSR of P\n");
        goto _free;
    }
    if (argc > 4 ){  //get the result matrix to check computations
        trgtMatrix = TMP_EXTRACTED_MARTIX;
        if (extractInTmpFS(argv[4],TMP_EXTRACTED_MARTIX) < 0)   trgtMatrix = argv[4];
        if (!( oracleOut = MMtoCSR(trgtMatrix) )){
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
        ret = EXIT_SUCCESS;
        goto _free;
    }
    //// PARALLEL COMPUTATIONs
#endif 

    if (!getConfig(&Conf)){
        VERBOSE printf("configuration changed from env");
    }
    int maxThreads = omp_get_max_threads();
    Conf.threadNum = (uint) maxThreads;
    DEBUG   printf("omp_get_max_threads:\t%d\n",maxThreads); 
    /*
     * get exported schedule configuration, 
     * if schedule != static -> dyn like -> set a chunk division function before omp for
     */
    int schedKind_chunk_monotonic[3];
    ompGetRuntimeSchedule(schedKind_chunk_monotonic);
    Conf.chunkDistrbFunc = &chunksNOOP; 
    if (schedKind_chunk_monotonic[0] != omp_sched_static)
        Conf.chunkDistrbFunc = chunkDistrbFunc;
    VERBOSE 
      printf("%s",Conf.chunkDistrbFunc == &chunksNOOP?"static schedule =>chunkDistrbFunc NOOP\n":"");
    //// PARALLEL COMPUTATIONs TO CHECK
    SP3GEMM_INTERF sp3gemm = &sp3gemmRowByRowPair;
    end = omp_get_wtime();elapsed = end-start;
    VERBOSE printf("preparing time: %le\t",elapsed);
    print3SPGEMMCore(R,AC,P,&Conf);
    ///test SP3GEMM as pair of SPGEMM: RAC = R * AC; RACP = RAC * P
    SPGEMM_INTERF spgemmFunc;
    for (uint f=0;  f<STATIC_ARR_ELEMENTS_N(SpgemmFuncs); f++){
        spgemmFunc = SpgemmFuncs[f];
        hprintsf("@computing Sp3GEMM as pair of SpGEMM with func:\%u at:%p\t",
          f,spgemmFunc);
        if (testSp3GEMMImpl(sp3gemm,spgemmFunc,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE printf("\nall SpgemmFuncs functions passed the test\n\n\n");
    ///test SP3GEMM as merged two multiplication
    for (uint f=0;  f<STATIC_ARR_ELEMENTS_N(Sp3gemmFuncs); f++){
        sp3gemm = Sp3gemmFuncs[f];
        hprintsf("@computing Sp3GEMM directly with func:\%u at:%p\t\t",
          f,sp3gemm);
        if (testSp3GEMMImpl(sp3gemm,NULL,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE printf("\nall Sp3gemmFuncs functions passed the test\n\n\n");

    DEBUGPRINT{
        printf("sparse matrix: AC_i\n");printSparseMatrix(outToCheck,TRUE);
    }
    
    ret = EXIT_SUCCESS;
    _free:
    if (R)          freeSpmat(R);
    if (AC)         freeSpmat(AC);
    if (P)          freeSpmat(P);
    if (oracleOut)  freeSpmat(oracleOut);
    if (outToCheck) freeSpmat(outToCheck);
    return ret;
}
