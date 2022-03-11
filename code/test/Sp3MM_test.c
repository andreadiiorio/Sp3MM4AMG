#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "sparseUtilsMulti.h"
#include "Sp3MM_CSR_OMP_Multi.h"
//#include "SpMMUtilsMulti.h"

#include "macros.h"
#include "utils.h"
#include "parser.h"
#include "ompChunksDivide.h"
#include "ompGetICV.h"  //ICV - RUNTIME information audit auxs

#ifdef CBLAS_TESTS
#include "SpMM_CBLAS.h"
#endif

////inline exports
spmat*  allocSpMatrix(ulong rows, ulong cols);
int     allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
spmat*  initSpMatrixSpMM(spmat* A, spmat* B);
void    freeSpmatInternal(spmat* mat);
void    freeSpmat(spmat* mat);
////inline exports
///multi implmentation functions
//void CAT(scSparseVectMul_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen, ACC_DENSE* aux);
//void CAT(scSparseVectMulPart_,OFF_F)(double scalar,double* vectVals,ulong* vectIdxs,ulong vectLen,ulong startIdx,ACC_DENSE* aux);
//void CAT(_scRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, ACC_DENSE* aux);
//void CAT(scSparseRowMul_,OFF_F)(double scalar,spmat* mat,ulong trgtR, ACC_DENSE* aux);
//ulong* CAT(spMMSizeUpperbound_,OFF_F)(spmat* A,spmat* B);

///single implmentation functions
//void freeSpMMAcc(SPMM_ACC* acc);
//void sparsifyDenseVect(SPMM_ACC* acc,ACC_DENSE* accV,SPACC* accSparse, ulong startColAcc);
int mergeRowsPartitions(SPACC* rowsParts,spmat* mat,CONFIG* conf);
//int mergeRows(SPACC* rows,spmat* mat);
//ACC_DENSE* _initAccVectors_monoalloc(ulong num,ulong size); //TODO PERF WITH NEXT
//int _allocAuxVect(ACC_DENSE* v,ulong size);
//void _resetAccVect(ACC_DENSE* acc);
//void _freeAccVectorsChecks(ACC_DENSE* vectors,ulong num); 
//void freeAccVectors(ACC_DENSE* vectors,ulong num);

void C_FortranShiftIdxs(spmat* outMat);
void Fortran_C_ShiftIdxs(spmat* m);


CHUNKS_DISTR    chunksFair,chunksFairFolded,chunksNOOP;

//global vars   ->  audit
CHUNKS_DISTR_INTERF chunkDistrbFunc=&chunksFairFolded;
//double Start,End,Elapsed,ElapsedInternal;
static CONFIG Conf = {
    .gridRows  = 8,
    .gridCols  = 8,
	.symbMMRowImplID = RBTREE,
};

/*
 * wrap result check and stats gather of Sp3MM implementation func at (@sp3mm,[@spmm])
 * result checked with @oracleOut or with CBLAS dense-serial implementation ifdef CBLAS_TESTS
 * @spmm forwarded to @sp3mm, stats printed on stdout; EXIT_FAILURE returned if any error
 */
static inline int testSp3MMImpl(SP3MM_INTERF sp3mm,SPMM_INTERF spmm,spmat* oracleOut,
  spmat* R,spmat* AC,spmat* P){
    int out = EXIT_FAILURE;
    spmat* outToCheck=NULL;
    //elapsed stats aux vars
    double times[AVG_TIMES_ITERATION],  timesInteral[AVG_TIMES_ITERATION];
    double deltaTStats[2],  deltaTInternalStats[2],internal_over_full,start,end;
    for (uint i=0;  i<AVG_TIMES_ITERATION; i++){
        start = omp_get_wtime();
        if (!(outToCheck = sp3mm(R,AC,P,&Conf,spmm))){
            ERRPRINTS("compute sp3mm at:%p with spmm at:%p failed...\n",sp3mm,spmm);
            return EXIT_FAILURE;
        }
        end = omp_get_wtime();

        #ifdef CBLAS_TESTS
        if (GEMMTripleCheckCBLAS(R,AC,P,outToCheck)){
             ERRPRINTS("compute sp3mm at:%p with spmm at:%p "
               "diff with CBLAS implementation...\n",sp3mm,spmm);
             goto _free;
        }
        #else
        if (oracleOut && spmatDiff(outToCheck,oracleOut))    goto _free;
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
"giving AC_{i+1} the outputs of the 3SPMM will be matched with the given result as an oracle" \
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
    } // else { //manually compute the correct result with a serial implementation
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
    print3SPMMCore(R,AC,P,&Conf);
    //// PARALLEL COMPUTATIONs TO CHECK
    end = omp_get_wtime();elapsed = end-start;
    VERBOSE printf("preparing time: %le\n",elapsed);
	
	//UB 		versions
    uint 			spMM_UB_FuncsN	 		= STATIC_ARR_ELEMENTS_N(Spmm_UB_Funcs_0);
	SPMM_INTERF*  	spMM_UB_Funcs  	 		= Spmm_UB_Funcs_0;
    SP3MM_INTERF  	sp3MM_UB_WrapPair 		= sp3mmRowByRowPair_0;	//compute sp3MM in 2 steps
    uint 			sp3MM_UB_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_UB_Funcs_0);
	SP3MM_INTERF* 	sp3MM_UB_Direct_Funcs	= Sp3mm_UB_Funcs_0;
	//symb-num	versions
    uint 			spMM_SymbNum_FuncsN	 	= STATIC_ARR_ELEMENTS_N(Spmm_SymbNum_Funcs_0);
	SPMM_INTERF*  	spMM_SymbNum_Funcs		= Spmm_SymbNum_Funcs_0;
    SP3MM_INTERF  	sp3MM_SymbNum_WrapPair	= sp3mmRowByRowPair_0;	//compute sp3MM in 2 steps
    uint 			sp3MM_SymbNum_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_SymbNum_Funcs_0);
	SP3MM_INTERF* 	sp3MM_SymbNum_Direct_Funcs	 = Sp3mm_SymbNum_Funcs_0;
	#ifdef MOCK_FORTRAN_INDEXING //test fortran integration
	//mock fortran app matrix passing shifting every nnz index
	//UB 		versions
    uint 			spMM_UB_FuncsN	 		= STATIC_ARR_ELEMENTS_N(Spmm_UB_Funcs_1);
	SPMM_INTERF*  	spMM_UB_Funcs  	 		= Spmm_UB_Funcs_1;
    SP3MM_INTERF  	sp3MM_UB_WrapPair 		= sp3mmRowByRowPair_1;	//compute sp3MM in 2 steps
    uint 			sp3MM_UB_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_UB_Funcs_1);
	SP3MM_INTERF* 	sp3MM_UB_Direct_Funcs	= Sp3mm_UB_Funcs_1;
	//symb-num	versions
    uint 			spMM_SymbNum_FuncsN	 	= STATIC_ARR_ELEMENTS_N(Spmm_SymbNum_Funcs_1);
	SPMM_INTERF*  	spMM_SymbNum_Funcs		= Spmm_SymbNum_Funcs_1;
    SP3MM_INTERF  	sp3MM_SymbNum_WrapPair	= sp3mmRowByRowPair_1;	//compute sp3MM in 2 steps
    uint 			sp3MM_SymbNum_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_SymbNum_Funcs_1);
	SP3MM_INTERF* 	sp3MM_SymbNum_Direct_Funcs	 = Sp3mm_SymbNum_Funcs_1;

	C_FortranShiftIdxs(R); C_FortranShiftIdxs(AC); C_FortranShiftIdxs(P);  
	if(oracleOut)	C_FortranShiftIdxs(oracleOut); 
	#endif
    SPMM_INTERF		spMMFunc;
	SP3MM_INTERF  	sp3MMFunc;

	//goto symbNum;	//TODO TODO
	ub:		///UB IMPLEMENTATIONS
	VERBOSE	hprintf("CHECKING UPPER BOUND IMPLEMENTATIONS\n");
    //test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
    for (uint f = 0;  f < spMM_UB_FuncsN; f++){
        spMMFunc = spMM_UB_Funcs[f];
        hprintsf("@computing Sp3MM as pair of SpMM with func:\%u at:%p\t",
          f,spMMFunc);
        if (testSp3MMImpl(sp3MM_UB_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM as merged two multiplication
    for (uint f = 0;  f < sp3MM_UB_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_UB_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly with func:\%u at:%p\t\t",
          f,sp3MMFunc);
        if (testSp3MMImpl(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE printf("\nall pairs of SpMM functions passed the test\n\n\n");
	symbNum:	///SYMB-NUM IMPLEMENTATIONS
	VERBOSE	hprintf("CHECKING SYMBOLIC.RBTREE - NUMERIC IMPLEMENTATIONS\n");
	Conf.symbMMRowImplID = RBTREE;
    //test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
    for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
        spMMFunc = spMM_SymbNum_Funcs[f];
        hprintsf("@computing Sp3MM as pair of SpMM with func:\%u at:%p\t",
          f,spMMFunc);
        if (testSp3MMImpl(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM as merged two multiplication
    for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly with func:\%u at:%p\t\t",
          f,sp3MMFunc);
        if (testSp3MMImpl(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
	VERBOSE	hprintf("CHECKING SYMBOLIC.IDXMAP - NUMERIC IMPLEMENTATIONS\n");
	Conf.symbMMRowImplID = IDXMAP;
    //test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
    for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
        spMMFunc = spMM_SymbNum_Funcs[f];
        hprintsf("@computing Sp3MM as pair of SpMM with func:\%u at:%p\t",
          f,spMMFunc);
        if (testSp3MMImpl(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM as merged two multiplication
    for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly with func:\%u at:%p\t\t",
          f,sp3MMFunc);
        if (testSp3MMImpl(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE hprintf("\nall sp3MM_SymbNum_Direct_Funcs functions passed the test\n\n\n");

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
