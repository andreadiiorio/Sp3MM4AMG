#ifndef SpGEMM_CBLAS_H
#define SpGEMM_CBLAS_H

/*
 * wrap CBLAS calls to compute @A * @B
 * where A: m x k   B: k x n
 */
double* dgemmCBLAS(double* A,double* B, uint m, uint n, uint k);

/*
 * check @outToCheck = @R * @AC * @P with CBLAS implementation
 */
int GEMMTripleCheckCBLAS(spmat* R,spmat* AC,spmat* P,spmat* RACP);

/*
 * check @AB = @A * @B with CBLAS implementation
 */
int GEMMCheckCBLAS(spmat* A,spmat* B,spmat* AB);

#endif
