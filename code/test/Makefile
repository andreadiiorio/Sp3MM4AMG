CC=gcc
FF=gfortran
CFLAGS=-Wall -Wextra -ggdb -O2
#libs
CFLAGS+=-lm
#TODO extra to reduce useless warnings
CFLAGS+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable
CFLAGS+=-Wno-unused-label
##TODO OMP CONFIG
#OMP_CANCELLATION=true
#export OMP_CANCELLATION

MACROS = -DDEBUGPRINT="if(FALSE)" -DDEBUG="if(FALSE)" -DVERBOSE="if(TRUE)" -DCONSISTENCY_CHECKS="if(FALSE)" -DDEBUGCHECKS="if(FALSE)"
CFLAGS+=$(RUNTIME)
#bind to source original project
srcDir=$(realpath ../src)

UNAME=$(shell uname -a | tr -c -d \[:alnum:\] | tr \[:lower:\] \[:upper:\] ) #upper uname-a
##CBLAS BUILD NEEDED
CBLAS_INCLUDE=CBLAS_LAPACK/include 
#EXTRA LIBS TO RELINK AFTER FORTRAN COMPILE -- SUPPORT DEBIAN - FEDORA
#EXTRALIBS=/usr/lib64/flexiblas/libflexiblas_openblas-openmp.so
ifeq ($(findstring DEBIAN,$(UNAME)),DEBIAN)
	EXTRALIBS =/usr/lib/x86_64-linux-gnu/libm.so.6
	EXTRALIBS+=/usr/lib/x86_64-linux-gnu/libgomp.so.1
	EXTRALIBS+=/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
	EXTRALIBS+=/usr/lib/x86_64-linux-gnu/libpthread.so.0
	EXTRALIBS+=/usr/lib/x86_64-linux-gnu/libc.so.6
	CFLAGS+= -fPIC #TODO NOT WORKING ON DELL FED BUILD.a
    STATIC_LIBS_DIR=lib_deb
else    #FEDORA
	EXTRALIBS =/usr/lib64/libm.so.6
	EXTRALIBS+=/usr/lib64/libgomp.so.1
	EXTRALIBS+=/usr/lib64/libgomp.so.1.0.0
	EXTRALIBS+=/usr/lib64/libpthread.so.0
	EXTRALIBS+=/usr/lib64/libc.so.6
    STATIC_LIBS_DIR=lib_fed
endif
CBLAS_SLIB=CBLAS_LAPACK/$(STATIC_LIBS_DIR)/libcblas.a
BLAS_SLIB=CBLAS_LAPACK/$(STATIC_LIBS_DIR)/librefblas.a

test_CBLAS_SpGEMM_OMP.elf: SpGEMM_OMP_test.c $(srcDir)/SpGEMM_OMP.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcDir)/include/*.h Makefile
	#create objectfile not linked
	$(CC) -O2 -fopenmp $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) $(filter-out %.h,$^) -c $(MACROS) -DTEST_MAIN
	#link it with static compiled cblas and source version library
	$(FF) -O2 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS)

test_CBLAS_SpGEMM_OMP_DBG.elf: SpGEMM_OMP_test.c $(srcDir)/SpGEMM_OMP.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcDir)/include/*.h Makefile
	#create objectfile not linked
	$(CC) -O0 -fopenmp $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) $(filter-out %.h,$^) -c  -DTEST_MAIN
	#link it with static compiled cblas and source version library
	$(FF) -O0 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS)

SpGEMM_OMP_with_test_CBLAS.elf: $(srcDir)/main_OMP.c $(srcDir)/SpGEMM_OMP.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcDir)/include/*.h SpGEMM_OMP_test.c 
	#create objectfile not linked
	$(CC) -fopenmp $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) -I. $(filter-out %.h,$^) -c $(MACROS) -DDEBUG_TEST_CBLAS
	#link it with static compiled cblas and source version library
	$(FF) -O2 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS) 

sparseMatrixToImage.elf:	sparseMatrixToImage.c $(srcDir)/commons/*.c $(srcDir)/lib/*.c $(srcDir)/include/*.h #TODO wiser selection of includes...
	$(CC) $(CFLAGS) -I$(srcDir)/include -I. $(filter-out %.h,$^) -o $@ $(MACROS) \
	-D MAIN_SPMAT_IMG -fopenmp
sparseMatrixToImageTest.elf:	sparseMatrixToImage.c $(srcDir)/commons/*.c $(srcDir)/lib/*.c $(srcDir)/include/*.h #TODO wiser selection of includes...
	$(CC) $(CFLAGS) -I$(srcDir)/include -I. $(filter-out %.h,$^) -o $@ $(MACROS) \
		-D MAIN_SPMAT_IMG -DTEST -O0 -fopenmp
testAll:
	$(shell cat ../../data/matrixGroups.list | xargs -L 1 sh -c 'echo $0 ${@};./test_CBLAS_SpGEMM_OMP.elf $0 ${@}' ) #TODO FIX
clean:
	rm -i *.elf

try:   
	@echo uname:        $(UNAME)
	@echo EXTRALIBS:    $(EXTRALIBS)
	
	
	
	

objs = test_SpGEMM_OMP.o sparseMatrixToImage.o
all: $(objs)
.PHONY: all clean testAll
