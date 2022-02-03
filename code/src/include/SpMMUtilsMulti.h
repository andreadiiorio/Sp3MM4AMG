#ifndef SPMMUTILSMULTI_H
#define SPMMUTILSMULTI_H

extern char TRGT_IMPL_START_IDX; //multi implementation switch

#ifdef OFF_F	//save "includer" OFF_F value before overwriting it
	#pragma push_macro("OFF_F")
	#define _OFF_F_OLD
	#undef  OFF_F
#endif

#define OFF_F 0
#include "SpMMUtilsGeneric.h"
#undef OFF_F

#define OFF_F 1
#include "SpMMUtilsGeneric.h"
#undef OFF_F

#ifdef _OFF_F_OLD
	#pragma pop_macro("OFF_F")
	#undef  _OFF_F_OLD
#endif

#endif

