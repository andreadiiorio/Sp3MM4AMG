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
#ifndef SpMM_CBLAS_H
#define SpMM_CBLAS_H

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
