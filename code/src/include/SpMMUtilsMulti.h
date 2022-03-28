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

