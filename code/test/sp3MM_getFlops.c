/*
 *    Sp3MM_for_AlgebraicMultiGrid
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

/*Simple program to get a baseline num of needed fp ops number for Sp3MM operation*/



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "Sp3MM_CSR_OMP_Multi.h"
#include "SpMMUtilsMulti.h"

#include "macros.h"
#include "utils.h"
#include "parser.h"

////inline exports
spmat *allocSpMatrix(ulong rows, ulong cols);
int allocSpMatrixInternal(ulong rows, ulong cols, spmat * mat);
spmat *initSpMatrixSpMM(spmat * A, spmat * B);
void freeSpmatInternal(spmat * mat);
void freeSpmat(spmat * mat);
int spVect_idx_in(idx_t idx, SPVECT_IDX_DENSE_MAP * idxsMapAcc);
void C_FortranShiftIdxs(spmat * outMat);
void Fortran_C_ShiftIdxs(spmat * m);

static CONFIG Conf = {
	.gridRows = 20,
	.gridCols = 2,
	.symbMMRowImplID = RBTREE,
};

void print3SPMMCore(spmat * R, spmat * AC, spmat * P, CONFIG * conf)
{
	printf("COARSENING AC: %lux%lu ---> %lux%lu\t"
	       "conf grid: %ux%u,\tNNZ:%lu-%lu-%lu\t AVG_TIMES_ITERATION:%u\t",
	       AC->M, AC->N, R->M, P->N, conf->gridRows, conf->gridCols,
	       R->NZ, AC->NZ, P->NZ, AVG_TIMES_ITERATION);
	printf("symbUBAssignType:%s\tbitmapLimbSize:%lu\n",
	       SPARSIFY_PRE_PARTITIONING ? "STATIC_ASSIGN" : "DYN_ASSIGN",
	       LIMB_SIZE_BIT);
}

/*
 * serial SpMM in output, 
 * returing C = A*B in out 
 * 		and in *@flopN the num of floating point operations
 */
spmat *spmmRowByRowGustavsonFlops(spmat * A, spmat * B, ulong * flopsN)
{
	spmat *AB = NULL;
	idx_t *rowsUBForecasts = NULL, *tmpJA;
	double *tmpAS;
	ACC_DENSE *acc = NULL;
	//matrix alloc
	if (!(AB = allocSpMatrix(A->M, B->N)))			goto _free;
	if (!(rowsUBForecasts = spMMSizeUpperbound_0(A, B)))	goto _free;
	ulong fullMatSizeUB = rowsUBForecasts[A->M];
	if (!(AB->JA = malloc(fullMatSizeUB * sizeof(*AB->JA)))) {
		ERRPRINT("spmmRowByRowGustavsonFlops JA malloc errd\n");
		goto _free;
	}
	if (!(AB->AS = malloc(fullMatSizeUB * sizeof(*AB->AS)))) {
		ERRPRINT("spmmRowByRowGustavsonFlops AS malloc errd\n");
		goto _free;
	}

	if (!(acc = malloc(sizeof(*acc)))) {
		ERRPRINT("spmmRowByRowGustavsonFlops acc malloc errd\n");
		goto _free;
	}
	if (allocAccDense(acc, B->N))				goto _free;
	idx_t irpCumul = 0;
	for (idx_t r = 0; r < A->M; r++) {
		for (ulong c = A->IRP[r]; c < A->IRP[r + 1]; c++) {	//row-by-row formul
			scSparseRowMul_0(A->AS[c], B, A->JA[c], acc);
		}
		irpCumul += acc->nnzIdxMap.len;
		AB->IRP[r + 1] = irpCumul;
		sparsifyDirect(acc, AB, r);	//0,NULL);TODO COL PARTITIONING COMMON API
		_resetAccVect(acc);
	}
	AB->NZ = irpCumul;
	*flopsN += fullMatSizeUB;
	//realloc out matrix to correct size
	if (!(tmpJA = realloc(AB->JA, AB->NZ * sizeof(*AB->JA)))) {
		ERRPRINT("spmmRowByRowGustavsonFlops realloc JA errd\n");
		goto _free;
	}
	AB->JA = tmpJA;
	if (!(tmpAS = realloc(AB->AS, AB->NZ * sizeof(*AB->AS)))) {
		ERRPRINT("spmmRowByRowGustavsonFlops realloc AS errd\n");
		goto _free;
	}
	AB->AS = tmpAS;

_free:
	if (acc)
		freeAccsDense(acc, 1);
	free(rowsUBForecasts);

	return AB;
}

#define HELP "usage Matrixes: R_{i+1}, AC_{i}, P_{i+1},[AC_{i+1}\n"
int main(int argc, char **argv)
{
	int ret = EXIT_FAILURE;
	if (init_urndfd())
		return ret;
	if (argc < 4) {
		ERRPRINT(HELP);
		return ret;
	}

	ulong flopN = 0;

	spmat *R = NULL, *AC = NULL, *P = NULL, *RAC = NULL, *RACP = NULL, 
	      *oracleOut = NULL;
	////parse sparse matrixes 
	char *trgtMatrix;
	trgtMatrix = TMP_EXTRACTED_MARTIX;
	if (extractInTmpFS(argv[1], TMP_EXTRACTED_MARTIX) < 0)
		trgtMatrix = argv[1];
	if (!(R = MMtoCSR(trgtMatrix))) {
		ERRPRINT("err during conversion MM -> CSR of R\n");
		goto _free;
	}
	trgtMatrix = TMP_EXTRACTED_MARTIX;
	if (extractInTmpFS(argv[2], TMP_EXTRACTED_MARTIX) < 0)
		trgtMatrix = argv[2];
	if (!(AC = MMtoCSR(trgtMatrix))) {
		ERRPRINT("err during conversion MM -> CSR of AC\n");
		goto _free;
	}
	trgtMatrix = TMP_EXTRACTED_MARTIX;
	if (extractInTmpFS(argv[3], TMP_EXTRACTED_MARTIX) < 0)
		trgtMatrix = argv[3];
	if (!(P = MMtoCSR(trgtMatrix))) {
		ERRPRINT("err during conversion MM -> CSR of P\n");
		goto _free;
	}
	if (argc > 4) {		//get the result matrix to check computations
		trgtMatrix = TMP_EXTRACTED_MARTIX;
		if (extractInTmpFS(argv[4], TMP_EXTRACTED_MARTIX) < 0)
			trgtMatrix = argv[4];
		if (!(oracleOut = MMtoCSR(trgtMatrix))) {
			ERRPRINT("err during conversion MM -> CSR AC_{i+1}\n");
			goto _free;
		}
	}
	CONSISTENCY_CHECKS {	///DIMENSION CHECKS...
		if (R->N != AC->M) {
			ERRPRINT("invalid sizes in R <-> AC\n");
			goto _free;
		}
		if (AC->N != P->M) {
			ERRPRINT("invalid sizes in AC <-> P\n");
			goto _free;
		}
		if (oracleOut) {
			if (R->M != oracleOut->M) {
				ERRPRINT("oracleOut invalid rows number\n");
				goto _free;
			}
			if (P->N != oracleOut->N) {
				ERRPRINT("oracleOut invalid cols number\n");
				goto _free;
			}
		}
	}
	DEBUGPRINT {
		printf("sparse matrix: R_{i+1}\n");
		printSparseMatrix(R, TRUE);
		printf("sparse matrix: AC_i\n");
		printSparseMatrix(AC, TRUE);
		printf("sparse matrix: P_{i+1}\n");
		printSparseMatrix(P, TRUE);
		if (argc > 4) {
			printf("sparse matrix: AC_{i+1}\n");
			printSparseMatrix(oracleOut, TRUE);
		}
	}

	printf("#%s %s %s %s\n", argv[1], argv[2], argv[3], argv[4]);
	print3SPMMCore(R, AC, P, &Conf);
	////Computations
	double times[AVG_TIMES_ITERATION];
	memset(times, 0, AVG_TIMES_ITERATION * sizeof(*times));
	double deltaTStats[2], deltaTInternalStats[2], notInternalTime, start, end;
	for (uint i = 0; i < AVG_TIMES_ITERATION; i++) {
		start = omp_get_wtime();
		if (!(RAC = spmmRowByRowGustavsonFlops(R, AC, &flopN)))
			goto _free;
		if (!(RACP = spmmRowByRowGustavsonFlops(RAC, P, &flopN)))
			goto _free;
		end = omp_get_wtime();
		assert(!oracleOut || !spmatDiff(RACP, oracleOut));
		times[i] = end - start;

		freeSpmat(RAC);
		RAC = NULL;
		freeSpmat(RACP);
		RACP = NULL;
	}
	statsAvgVar(times, AVG_TIMES_ITERATION, deltaTStats);
	printf("Sp3MM as 2 SpMM\tflop:%lu\telapsedAvg:%lf\telapsedVar:%lf\tMegaflopsAvg:%lf\n",
	     flopN, deltaTStats[0], deltaTStats[1], flopN / (deltaTStats[0] * 1e6));

	ret = EXIT_SUCCESS;

	_free:
	freeSpmat(R);
	freeSpmat(AC);
	freeSpmat(P);
	freeSpmat(RAC);
	freeSpmat(RACP);
	freeSpmat(oracleOut);

	return ret;
}
