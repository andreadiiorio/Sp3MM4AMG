//Get multiple implementation for C-Fortran indexing by re-define & re-include

#define OFF_F 0
#include "Sp3MM_CSR_OMP_Generic.c"
#undef OFF_F

#define OFF_F 1
#include "Sp3MM_CSR_OMP_Generic.c"
#undef OFF_F
