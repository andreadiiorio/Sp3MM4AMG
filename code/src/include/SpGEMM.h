#ifndef SPGEMM
#define SPGEMM

//compute function interface and its pointer definitions
typedef int ( COMPUTEFUNC       ) (spmat*,spmat*,spmat*,CONFIG*,spmat*);
typedef int (*COMPUTEFUNC_INTERF) (spmat*,spmat*,spmat*,CONFIG*,spmat*);
/*
 * basic spgemm with row partitioning, 1 row per thread in consecutive order
 * statically assigned to threads
 */
int spgemmRowsBasic(spmat* R,spmat* AC,spmat* P,CONFIG*, spmat* out);
#endif
