#ifndef SP3MM_CSR_OMP_SYMB_MULTI
#define SP3MM_CSR_OMP_SYMB_MULTI

/* Multi implementation of symbolic product of sparse matrixes, config macros
 * OFF_F:		C-Fortran spmat indexing
 * OUT_IDXS:	indexes output
 * COL_PARTS:	partitioning columns output...
 */

#define OUT_IDXS_ON		OutIdxs_
#define COL_PARTS_ON	ColParts_
#undef OUT_IDXS
#undef COL_PARTS

#define OFF_F 0
	///generate basic versions
	#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
	///generate outIdxs versions
	#define OUT_IDXS 	OUT_IDXS_ON	
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
	#undef  OUT_IDXS
	///generate colParts versions
	#define COL_PARTS	COL_PARTS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
		//generate outIdxs AND colParts ve sions
		#define OUT_IDXS 	OUT_IDXS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"

	#undef OUT_IDXS
	#undef COL_PARTS
#undef OFF_F
#define OFF_F 1
	///generate basic versions
	#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
	///generate outIdxs versions
	#define OUT_IDXS 	OUT_IDXS_ON	
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
	#undef  OUT_IDXS
	///generate colParts versions
	#define COL_PARTS	COL_PARTS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"
		//generate outIdxs AND colParts ve sions
		#define OUT_IDXS 	OUT_IDXS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.h"

	#undef OUT_IDXS
	#undef COL_PARTS
#undef OFF_F

#endif
