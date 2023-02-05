/*
 *              Sp3MM_for_AlgebraicMultiGrid
 *    (C) Copyright 2021-2022
 *        Andrea Di Iorio      
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions, and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the Sp3MM_for_AlgebraicMultiGrid or the names of its contributors may
 *       not be used to endorse or promote products derived from this
 *       software without specific written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 *  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE Sp3MM_for_AlgebraicMultiGrid GROUP OR ITS CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
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


////inline exports
spmat*  allocSpMatrix(ulong rows, ulong cols);
int	 allocSpMatrixInternal(ulong rows, ulong cols, spmat* mat);
spmat*  initSpMatrixSpMM(spmat* A, spmat* B);
void	freeSpmatInternal(spmat* mat);
void	freeSpmat(spmat* mat);
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


CHUNKS_DISTR	chunksFair,chunksFairFolded,chunksNOOP;

//global vars   ->  audit
CHUNKS_DISTR_INTERF chunkDistrbFunc=&chunksFairFolded;
//double Start,End,Elapsed,ElapsedInternal;
static CONFIG Conf = {
	.gridRows  = 20,
	.gridCols  = 2,
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
 * result checked with @oracleOut 
 * @spmm forwarded to @sp3mm, stats printed on stdout; EXIT_FAILURE returned if any error
 */
static inline int testSp3MMImplOMP(SP3MM_INTERF sp3mm,SPMM_INTERF spmm,spmat* oracleOut,
				   spmat* R,spmat* AC,spmat* P){
	int out = EXIT_FAILURE;
	spmat* outToCheck=NULL;
	//elapsed stats aux vars
	double times[AVG_TIMES_ITERATION],  timesInteral[AVG_TIMES_ITERATION];
	memset(times,0,AVG_TIMES_ITERATION*sizeof(*times));
	memset(timesInteral,0,AVG_TIMES_ITERATION*sizeof(*timesInteral));

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

			if (oracleOut && spmatDiff(outToCheck,oracleOut))	goto _free;

			freeSpmat(outToCheck); outToCheck = NULL;
			times[i]		= end - start;
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

//see next function
#if !defined FIXED_2D_PARTITIONING && !defined MPI_CART_DIMFACT_DIVISION_2D_PARTITIONING && !defined OVERFIT_GRIDROWS_GIVEN_COLS_2D_PARTITIONING
	#define MPI_CART_DIMFACT_DIVISION_2D_PARTITIONING
#endif
#if defined FIXED_2D_PARTITIONING && (!defined FIXED_2D_PARTITIONING_ROWS || !defined FIXED_2D_PARTITIONING_ROWS)
	#define FIXED_2D_PARTITIONING_ROWS 8
	#define FIXED_2D_PARTITIONING_COLS 8
#endif //FIXED_2D_PARTITIONING
#ifndef SPMM_1DBLOCKS_THREAD_ITERATION_FACTOR
	#define SPMM_1DBLOCKS_THREAD_ITERATION_FACTOR	2
#endif //SPMM_1DBLOCKS_THREAD_ITERATION_FACTOR
/*
 * adapt omp parallelization grid with respect to the current SpMM implementation (*not Sp3MM*)
 * @fID, with :
 *	0 -> SpMM 1D direct
 *	1 -> SpMM 1D Blocks
 *	2 -> SpMM 2D offsets
 *	3 -> SpMM 2D allocated
 */
static inline void _adaptGridByImpl(uint fID,const ushort ORIG_GRID_CONF[2]){
	int dims[2] = {0,0};
	switch(fID){
		case _1D_DIRECT:	return;
		case _1D_BLOCKS:
			Conf.gridRows = Conf.threadNum * SPMM_1DBLOCKS_THREAD_ITERATION_FACTOR;
			break;
		case _2D_OFFSET:
		case _2D_ALLOCD:
			#if		defined	FIXED_2D_PARTITIONING
			Conf.gridRows = FIXED_2D_PARTITIONING_ROWS
			Conf.gridCols = FIXED_2D_PARTITIONING_COLS
			#elif	defined	MPI_CART_DIMFACT_DIVISION_2D_PARTITIONING
			if(MPI_Dims_create(Conf.threadNum,2,dims)) {ERRPRINT("MPIDimscreate ERRD!!!\n");exit(55);}
			if(dims[1] == 1){ 
				//threadNum is prime ... not fully dividable... add some reminder dividing threadNum+1
				if(MPI_Dims_create(Conf.threadNum+1,2,dims)){ERRPRINT("MPIDimscreate ERRD!!!\n");exit(55);}
			}
			Conf.gridRows = dims[0];
			Conf.gridCols = dims[1];
			#elif	defined	OVERFIT_GRIDROWS_GIVEN_COLS_2D_PARTITIONING
			Conf.gridCols = ORIG_GRID_CONF[1];
			Conf.gridRows = INT_DIV_CEIL(Conf.threadNum,Conf.gridCols);
			#endif	//defined FIXED_2D_PARTITIONING

	}
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
	CONSISTENCY_CHECKS{	 ///DIMENSION CHECKS...
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
		printf("sparse matrix: AC_i\n");	printSparseMatrix(AC,TRUE);
		printf("sparse matrix: P_{i+1}\n"); printSparseMatrix(P,TRUE);
		if (argc > 4){ 
			printf("sparse matrix: AC_{i+1}\n");printSparseMatrix(oracleOut,TRUE);
		}
	}

	if (!getConfig(&Conf)){
		VERBOSE printf("configuration changed from env");
	}
	//save exported parallelization grid configured ... see Config.md in the top folder
	//const ushort GRIDROWS = Conf.gridRows;
	//const ushort GRIDCOLS = Conf.gridCols;
	const ushort ORIG_GRID_CONF[2] = {Conf.gridRows,Conf.gridCols};

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
	uint 		spMM_UB_FuncsN		= STATIC_ARR_ELEMENTS_N(Spmm_UB_Funcs_0);
	SPMM_INTERF*  	spMM_UB_Funcs  	 	= Spmm_UB_Funcs_0;
	SP3MM_INTERF  	sp3MM_UB_WrapPair 	= sp3mmRowByRowPair_0;	//compute sp3MM in 2 steps
	uint 		sp3MM_UB_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_UB_Funcs_0);
	SP3MM_INTERF* 	sp3MM_UB_Direct_Funcs	= Sp3mm_UB_Funcs_0;
	//symb-num	versions
	uint 		spMM_SymbNum_FuncsN	= STATIC_ARR_ELEMENTS_N(Spmm_SymbNum_Funcs_0);
	SPMM_INTERF*  	spMM_SymbNum_Funcs	= Spmm_SymbNum_Funcs_0;
	SP3MM_INTERF  	sp3MM_SymbNum_WrapPair	= sp3mmRowByRowPair_0;	//compute sp3MM in 2 steps
	uint 		sp3MM_SymbNum_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_SymbNum_Funcs_0);
	SP3MM_INTERF* 	sp3MM_SymbNum_Direct_Funcs   = Sp3mm_SymbNum_Funcs_0;
	#ifdef MOCK_FORTRAN_INDEXING //test fortran integration
	//mock fortran app matrix passing shifting every nnz index
	//UB 		versions
	spMM_UB_FuncsN	 	= STATIC_ARR_ELEMENTS_N(Spmm_UB_Funcs_1);
	spMM_UB_Funcs  	 	= Spmm_UB_Funcs_1;
	sp3MM_UB_WrapPair 	= sp3mmRowByRowPair_1;	//compute sp3MM in 2 steps
	sp3MM_UB_Direct_FuncsN  = STATIC_ARR_ELEMENTS_N(Sp3mm_UB_Funcs_1);
	sp3MM_UB_Direct_Funcs	= Sp3mm_UB_Funcs_1;
	//symb-num	versions
	spMM_SymbNum_FuncsN	= STATIC_ARR_ELEMENTS_N(Spmm_SymbNum_Funcs_1);
	spMM_SymbNum_Funcs	= Spmm_SymbNum_Funcs_1;
	sp3MM_SymbNum_WrapPair	= sp3mmRowByRowPair_1;	//compute sp3MM in 2 steps
	sp3MM_SymbNum_Direct_FuncsN	= STATIC_ARR_ELEMENTS_N(Sp3mm_SymbNum_Funcs_1);
	sp3MM_SymbNum_Direct_Funcs	= Sp3mm_SymbNum_Funcs_1;

	C_FortranShiftIdxs(R); C_FortranShiftIdxs(AC); C_FortranShiftIdxs(P);  
	if(oracleOut)	C_FortranShiftIdxs(oracleOut); 
	#endif
	SPMM_INTERF		spMMFunc;
	SP3MM_INTERF  	sp3MMFunc;

	//goto symbNum_idxMap;	//TODO TODO
	
	//ub:		///UB IMPLEMENTATIONS
	VERBOSE	hprintf("CHECKING UPPER BOUND IMPLEMENTATIONS\n");
	//test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
	for (uint f = 0;  f < spMM_UB_FuncsN; f++){
		spMMFunc = spMM_UB_Funcs[f];
		hprintsf("@computing Sp3MM as pair of SpMM UpperBounded \tfunc:\%u at:%p\n",f,spMMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
		if (testSp3MMImplOMP(sp3MM_UB_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
	}
	//test SP3MM directly as merged two multiplication
	for (uint f = 0;  f < sp3MM_UB_Direct_FuncsN; f++){
		sp3MMFunc = sp3MM_UB_Direct_Funcs[f];
		hprintsf("@computing Sp3MM directly UpperBounded \tfunc:\%u at:%p\t\n",f,sp3MMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
		if (testSp3MMImplOMP(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
	}
	VERBOSE printf("\nall pairs of SpMM functions passed the test\n\n\n");
	#ifdef	 UB_IMPL_ONLY
	ret = EXIT_SUCCESS;
	goto _free;
	#endif	//UB_IMPL_ONLY
	///SYMB-NUM IMPLEMENTATIONS
	//symbNum:
	//symbNum_rbtree:
	VERBOSE	hprintf("CHECKING SYMBOLIC.RBTREE - NUMERIC IMPLEMENTATIONS\n");
	Conf.symbMMRowImplID = RBTREE;
	//test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
	for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
		spMMFunc = spMM_SymbNum_Funcs[f];
		hprintsf("@computing Sp3MM as pair of SpMM SymbolicAccurate with RBTREE \tfunc:\%u at:%p\n",
		  f,spMMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
		if (testSp3MMImplOMP(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
	}
	//test SP3MM directly as merged two multiplication
	for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
		sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
		hprintsf("@computing Sp3MM directly SymbolicAccurate with RBTREE \tfunc:\%u at:%p\t\n",
		  f,sp3MMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
		if (testSp3MMImplOMP(sp3MMFunc,NULL,oracleOut,R,AC,P))   goto _free;
	}
	VERBOSE	hprintf("CHECKING SYMBOLIC.IDXMAP - NUMERIC IMPLEMENTATIONS\n");
	//symbNum_idxMap:
	Conf.symbMMRowImplID = IDXMAP;
	//test SP3MM as pair of SPMM: RAC = R * AC; RACP = RAC * P
	for (uint f = 0;  f < spMM_SymbNum_FuncsN; f++){
		spMMFunc = spMM_SymbNum_Funcs[f];
		hprintsf("@computing Sp3MM as pair of SpMM SymbolicAccurate with IDXMAP \tfunc:\%u at:%p\n",
		  f,spMMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
		if (testSp3MMImplOMP(sp3MM_SymbNum_WrapPair,spMMFunc,oracleOut,R,AC,P))   goto _free;
	}
	//test SP3MM as merged two multiplication
	for (uint f = 0;  f < sp3MM_SymbNum_Direct_FuncsN; f++){
		sp3MMFunc = sp3MM_SymbNum_Direct_Funcs[f];
		hprintsf("@computing Sp3MM directly with IDXMAP \tfunc:\%u at:%p\t\n",
		  f,sp3MMFunc);
		_adaptGridByImpl(f,ORIG_GRID_CONF);
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
	freeSpmat(R);
	freeSpmat(AC);
	freeSpmat(P);
	freeSpmat(oracleOut);
	freeSpmat(outToCheck);
	return ret;
}
