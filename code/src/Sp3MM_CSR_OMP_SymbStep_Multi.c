#include "macros.h"
#include "config.h"
#include "sparseMatrix.h"
#include "linuxK_rbtree_minimalized.h"

#include "Sp3MM_CSR_OMP_SymbStep_Multi.h"

#include "sparseUtilsMulti.h"
#include "utils.h"
#include <omp.h>

//inline exports
//ushort matchingUnifRangeIdx(idx_t idx, idx_t size, ushort rangesN);
idx_t* CAT(spMMSizeUpperbound_,0)(spmat* ,spmat* );
idx_t* CAT(spMMSizeUpperbound_,1)(spmat* ,spmat* );
idx_t* CAT(spMMSizeUpperboundColParts_,0)(spmat* ,spmat* ,ushort);
idx_t* CAT(spMMSizeUpperboundColParts_,1)(spmat* ,spmat* ,ushort);

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
	#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
	///generate outIdxs versions
	#define OUT_IDXS 	OUT_IDXS_ON	
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
	#undef  OUT_IDXS
	///generate colParts versions
	#define COL_PARTS	COL_PARTS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
		//generate outIdxs AND colParts versions
		#define OUT_IDXS 	OUT_IDXS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"

	#undef OUT_IDXS
	#undef COL_PARTS
#undef OFF_F
#define OFF_F 1
	///generate basic versions
	#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
	///generate outIdxs versions
	#define OUT_IDXS 	OUT_IDXS_ON	
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
	#undef  OUT_IDXS
	///generate colParts versions
	#define COL_PARTS	COL_PARTS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"
		//generate outIdxs AND colParts versions
		#define OUT_IDXS 	OUT_IDXS_ON
		#include "Sp3MM_CSR_OMP_SymbStep_Generic.c"

	#undef OUT_IDXS
	#undef COL_PARTS
#undef OFF_F
