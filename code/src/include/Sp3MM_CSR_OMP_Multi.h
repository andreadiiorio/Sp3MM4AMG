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
//CORE IMPLEMENTATIONS HEADER
#ifndef SP3MM_CSR_OMP_MULTI_H
#define SP3MM_CSR_OMP_MULTI_H

///commons single implementation stuff
#include "macros.h"
#include "sparseMatrix.h"

///aux structures
//hold SPMM result over a unpartitionated space among threads-row[s' blocks]
typedef struct{
    //space to hold SPMM output
    ulong*  JA;
    double* AS;
    ulong   size;			//num of entries allocated -> only dbg checks
    ulong   lastAssigned;	//last JA&AS assigned index to an accumulator(atom)
    SPACC*  accs;			//SPARSIFIED ACC POINTERS
	uint	accsNum;	
} SPMM_ACC; //accumulator for SPMM
///compute function interface and its pointer definitions
typedef spmat* ( SPMM        )  (spmat*,spmat*,CONFIG*);
typedef spmat* (*SPMM_INTERF )  (spmat*,spmat*,CONFIG*);
typedef spmat* ( SP3MM       )  (spmat*,spmat*,spmat*,CONFIG*,SPMM_INTERF);
typedef spmat* (*SP3MM_INTERF)  (spmat*,spmat*,spmat*,CONFIG*,SPMM_INTERF);
///-- commons single implementation stuff

///includes
#include "linuxK_rbtree_minimalized.h"
#include "Sp3MM_CSR_OMP_SymbStep_Multi.h"

//extern char TRGT_IMPL_START_IDX; //multi implementation switch
#include "sparseUtilsMulti.h"
#ifdef OFF_F	//save "includer" OFF_F value before overwriting it
	#pragma push_macro("OFF_F")
	#define _OFF_F_OLD
	#undef  OFF_F
#endif


#define OFF_F 0
#include "Sp3MM_CSR_OMP_UB_Generic.h"
#include "Sp3MM_CSR_OMP_Symb_Generic.h"
#undef OFF_F

#define OFF_F 1
#include "Sp3MM_CSR_OMP_UB_Generic.h"
#include "Sp3MM_CSR_OMP_Symb_Generic.h"
#undef OFF_F



#ifdef _OFF_F_OLD
	#pragma pop_macro("OFF_F")
	#undef  _OFF_F_OLD
#endif


#endif	//SP3MM_CSR_OMP_MULTI_H 
