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
