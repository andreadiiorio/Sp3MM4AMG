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
//Get multiple implementation for C-Fortran indexing by re-define & re-include
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <alloca.h> //TODO quick hold few CSR cols partition sizes
#include <omp.h>
//UB version deps
#include "Sp3MM_CSR_OMP_Multi.h"
#include "SpMMUtilsMulti.h"
#include "sparseUtilsMulti.h"
#include "ompChunksDivide.h"
#include "parser.h"
#include "utils.h"
#include "macros.h"
#include "sparseMatrix.h"

#include "inlineExports.h"

//Symb version deps
#include "Sp3MM_CSR_OMP_Multi.h"


//global vars	->	audit
double Start,End,Elapsed,ElapsedInternal;

#define OFF_F 0
#include "inlineExports_Generic.h"
#include "Sp3MM_CSR_OMP_UB_Generic.c"
#include "Sp3MM_CSR_OMP_Symb_Generic.c"
#undef OFF_F

#define OFF_F 1
#include "inlineExports_Generic.h"
#include "Sp3MM_CSR_OMP_UB_Generic.c"
#include "Sp3MM_CSR_OMP_Symb_Generic.c"
#undef OFF_F
