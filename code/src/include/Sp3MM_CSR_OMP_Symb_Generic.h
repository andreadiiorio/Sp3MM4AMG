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

#ifndef OFF_F
    //#pragma message("generic implementation requires OFF_F defined")
    #error generic implementation requires OFF_F defined
#endif

/////SYMBOLIC - NUMERIC IMPLEMENTATIONS
///SP3MM FUNCTIONS
/*
 *  triple matrix multiplication among @R * @AC * @P using gustavson parallel implementation
 *  implmented as a pair of subsequent spmm operations
 *  if @conf->spmm != NULL, it will be used as spmm function, otherwise euristics will be 
 *  used to decide wich implementation to use
 */
SP3MM CAT(sp3mmRowByRowPair_SymbNum_,OFF_F);

/*
 * row-by-row-by-row implementation: forwarding @R*@AC rth row to P for row-by-row
 * accumulation in preallocated space, TODO exactly determined
 * basic parallelization: 1thread per @R's rows that will also forward the result to P
 */
SP3MM CAT(sp3mmRowByRowMerged_SymbNum_,OFF_F);

///SUB FUNCTIONS
///SPMM FUNCTIONS
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson row-by-row
 * formulation using an aux dense vector @_auxDense
 * return resulting product matrix
 */
SPMM CAT(spmmRowByRow_SymbNum_,OFF_F);
/*
 * sparse parallel implementation of @A * @B parallelizing Gustavson 
 * with partitioning of @A in @conf->gridRows blocks of rows  
 * return resulting product matrix
 */
SPMM CAT(spmmRowByRow1DBlocks_SymbNum_,OFF_F);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, accessed by aux offsets
 */
SPMM CAT(spmmRowByRow2DBlocks_SymbNum_,OFF_F);

/* 
 * sparse parallel implementation of @A * @B as Gustavson parallelizzed in 2D
 * with partitioning of
 * @A into rows groups, uniform rows division
 * @B into cols groups, uniform cols division, ALLOCATED as CSR submatrixes
 */
SPMM CAT(spmmRowByRow2DBlocksAllocated_SymbNum_,OFF_F);

///implementation wrappers as static array of function pointers 
//sp3mm as pair of spmm
static SPMM_INTERF  CAT(Spmm_SymbNum_Funcs_,OFF_F)[] = {
    & CAT(spmmRowByRow_SymbNum_,OFF_F),
    & CAT(spmmRowByRow1DBlocks_SymbNum_,OFF_F),
    //& CAT(spmmRowByRow2DBlocks_SymbNum_,OFF_F),
    //& CAT(spmmRowByRow2DBlocksAllocated_SymbNum_,OFF_F)
};
//sp3mm as pair of spmm
static SP3MM_INTERF CAT(Sp3mm_SymbNum_Funcs_,OFF_F)[] = {
    //& CAT(sp3mmRowByRowMerged_SymbNum_,OFF_F)
};