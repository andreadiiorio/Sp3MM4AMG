CC=gcc
FF=gfortran
CFLAGS=-Wall -Wextra -ggdb -O2
#libs
CFLAGS+=-lm
#TODO extra to reduce useless warnings
CFLAGS+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable
MACROS = -DDEBUGPRINT="if(FALSE)" -DVERBOSE="if(FALSE)"

#bind to source original project
srcDir=$(realpath ../src)
##CBLAS BUILD NEEDED
CBLAS_INCLUDE=CBLAS_LAPACK/include 
CBLAS_SLIB=CBLAS_LAPACK/lib/libcblas.a
BLAS_SLIB=CBLAS_LAPACK/lib/librefblas.a
#EXTRA LIBS
#EXTRALIBS=/usr/lib64/flexiblas/libflexiblas_openblas-openmp.so
EXTRALIBS =/usr/lib64/libm.so.6
EXTRALIBS+=/usr/lib64/libgomp.so.1
EXTRALIBS+=/usr/lib64/libgomp.so.1.0.0
EXTRALIBS+=/usr/lib64/libpthread.so.0
EXTRALIBS+=/usr/lib64/libc.so.6

test_CBLAS_SpGEMM_OMP.elf: SpGEMM_OMP_test.c $(srcDir)/SpGEMM_OMP.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcDir)/include/*.h
	@rm -f *.o
	#create objectfile not linked
	$(CC) -fopenmp $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) $(filter-out %.h,$^) -c $(MACROS) -DTEST_MAIN
	#link it with static compiled cblas and source version library
	$(FF) -O2 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS)

SpGEMM_OMP_with_test_CBLAS.elf: $(srcDir)/main_OMP.c $(srcDir)/SpGEMM_OMP.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcDir)/include/*.h SpGEMM_OMP_test.c 
	#create objectfile not linked
	$(CC) -fopenmp $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) -I. $(filter-out %.h,$^) -c $(MACROS) -DDEBUG_TEST_CBLAS
	#link it with static compiled cblas and source version library
	$(FF) -O2 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS)

clean:
	rm -i *.elf

objs = test_SpGEMM_OMP.o
all: $(objs)
.PHONY: all
