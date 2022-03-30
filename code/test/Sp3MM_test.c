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

void print3SPMMCore(spmat* R,spmat* AC,spmat* P,CONFIG* conf){
    printf("COARSENING AC: %lux%lu ---> %lux%lu\t"
      "conf grid: %ux%u,\tNNZ:%lu-%lu-%lu\t AVG_TIMES_ITERATION:%u\t",
      AC->M,AC->N, R->M,P->N, conf->gridRows,conf->gridCols,
      R->NZ,AC->NZ,P->NZ,AVG_TIMES_ITERATION);
	printf("symbUBAssignType:%s\tbitmapLimbSize:%lu\n",
	  SPARSIFY_PRE_PARTITIONING?"STATIC_ASSIGN":"DYN_ASSIGN",LIMB_SIZE_BIT);
}

/*
 * wrap result check and stats gather of Sp3MM implementation func at (@sp3mm,[@spmm])
 * result checked with @oracleOut or with CBLAS dense-serial implementation ifdef CBLAS_TESTS
 * @spmm forwarded to @sp3mm, stats printed on stdout; EXIT_FAILURE returned if any error
 */
static inline int testSp3MMImplOMP(SP3MM_INTERF sp3mm,SPMM_INTERF spmm,spmat* oracleOut,
  spmat* R,spmat* AC,spmat* P){
    int out = EXIT_FAILURE;
    spmat* outToCheck=NULL;
    //elapsed stats aux vars
    double times[AVG_TIMES_ITERATION],  timesInteral[AVG_TIMES_ITERATION];
    double deltaTStats[2],  deltaTInternalStats[2],notInternalTime,start,end;
	uint threadNum = Conf.threadNum;
	#ifdef DECREASE_THREAD_NUM
    for (uint t=Conf.threadNum;  t>0; t--){
		omp_set_num_threads(t);
		threadNum = t;
	#endif
    	for (uint i=0;  i<AVG_TIMES_ITERATION; i++){
    	    start = omp_get_wtime();
    	    if (!(outToCheck = sp3mm(R,AC,P,&Conf,spmm))){
    	        ERRPRINTS("compute sp3mm at:%p with spmm at:%p failed...\n",sp3mm,spmm);
    	        return EXIT_FAILURE;
    	    }
    	    end = omp_get_wtime();

    	    if (oracleOut && spmatDiff(outToCheck,oracleOut))    goto _free;

    	    freeSpmat(outToCheck); outToCheck = NULL;
    	    times[i]        = end - start;
    	    timesInteral[i] = ElapsedInternal;
    	    ElapsedInternal = Elapsed = 0;
    	}
    	statsAvgVar(times,AVG_TIMES_ITERATION,deltaTStats);
    	statsAvgVar(timesInteral,AVG_TIMES_ITERATION,deltaTInternalStats);
    	notInternalTime = 1 - deltaTInternalStats[0] / deltaTStats[0];
    	printf("threadNum: %d\tompGridSize: %ux%u\t"
    	 "timeAvg:%le timeVar:%le\ttimeInternalAvg:%le (overheads ~ %lf%% tot) timeInternalVar:%le \n",
    	 threadNum,Conf.gridRows,Conf.gridCols,
		 deltaTStats[0],deltaTStats[1],deltaTInternalStats[0],notInternalTime*100,deltaTInternalStats[1]);
	#ifdef DECREASE_THREAD_NUM
	}
	#endif
    
    out = EXIT_SUCCESS;
    _free:
    if(outToCheck)  freeSpmat(outToCheck);
    return out;
}

#define TESTTESTS "TESTTESTS"
#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},[AC_{i+1} || " TESTTESTS "(requires -DCBLAS_TEST)]"\
"all matriexes in MatrixMarket_sparse_matrix_COO[.compressed] and the triple product will be matched with an oracle output\n" \
"giving AC_{i+1} the oracle output will be from the given input, otherwise it will be computed with a pair of SpMM as serial implementations" \
"\ngiving" TESTTESTS "the oracle output from the serial implementation will be validated with CBLAS netlib reference implementation (will densify)"


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
    } else {
		if (!(oracleOut = sp3mmRowByRowPair_0(R,AC,P,&Conf,&spmmSerial_0))){
			ERRPRINT("err during sp3mm for oracle\n");
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
    ////Validate the serialImplementation with CBLAS reference implementation
    if ( argc == 4 && !strncmp(argv[5],TESTTESTS,strlen(TESTTESTS)) ){
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
    printf("#%s %s %s %s\n", argv[1],argv[2],argv[3],argv[4]);
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
        hprintsf("@computing Sp3MM as pair of SpMM UpperBounded \tfunc:\%u at:%p\n",
          f,spMMFunc);
        if (testSp3MMImplOMP(sp3MM_UB_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM directly as merged two multiplication
    for (uint f = 0;  f < sp3MM_UB_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_UB_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly UpperBounded \tfunc:\%u at:%p\t\n",
          f,sp3MMFunc);
        if (testSp3MMImplOMP(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE printf("\nall pairs of SpMM functions passed the test\n\n\n");
	///SYMB-NUM IMPLEMENTATIONS
	symbNum:
	VERBOSE	hprintf("CHECKING SYMBOLIC.RBTREE - NUMERIC IMPLEMENTATIONS\n");
	Conf.symbMMRowImplID = RBTREE;
    //test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
    for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
        spMMFunc = spMM_SymbNum_Funcs[f];
        hprintsf("@computing Sp3MM as pair of SpMM SymbolicAccurate with RBTREE \tfunc:\%u at:%p\n",
          f,spMMFunc);
        if (testSp3MMImplOMP(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM directly as merged two multiplication
    for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly SymbolicAccurate with RBTREE \tfunc:\%u at:%p\t\n",
          f,sp3MMFunc);
        if (testSp3MMImplOMP(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
	VERBOSE	hprintf("CHECKING SYMBOLIC.IDXMAP - NUMERIC IMPLEMENTATIONS\n");
	Conf.symbMMRowImplID = IDXMAP;
    //test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
    for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
        spMMFunc = spMM_SymbNum_Funcs[f];
        hprintsf("@computing Sp3MM as pair of SpMM with IDXMAP \tfunc:\%u at:%p\n",
          f,spMMFunc);
        if (testSp3MMImplOMP(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
    }
    //test SP3MM as merged two multiplication
    for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
        sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
        hprintsf("@computing Sp3MM directly with IDXMAP \tfunc:\%u at:%p\t\n",
          f,sp3MMFunc);
        if (testSp3MMImplOMP(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
    }
    VERBOSE hprintf("\nall Sp3MM_SymbNum functions passed the test\n\n\n");
	//test wraps end
    VERBOSE hprintf("\nall Sp3MM functions passed the test\n\n\n");

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
