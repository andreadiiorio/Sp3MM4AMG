#              Sp3MM_for_AlgebraicMultiGrid
#    (C) Copyright 2021-2022
#        Andrea Di Iorio      
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions, and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. The name of the Sp3MM_for_AlgebraicMultiGrid or the names of its contributors may
#       not be used to endorse or promote products derived from this
#       software without specific written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE Sp3MM_for_AlgebraicMultiGrid GROUP OR ITS CONTRIBUTORS
#  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

CC=gcc
FF=gfortran
#libs
#extra to reduce useless warnings
CWALL =-Wall -Wextra
CWALL+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-switch
CWALL+=-Wno-unused-function -Wno-unused-variable       #TODO CLEAN USELESS FUNCTIONS / VARIABLES
CWALL+=-Wno-unused-label -Wfatal-errors
CFLAGS=-O2 $(CWALL)
CFLAGS+=-lm -fopenmp

#SYSTEM CONFIGURATION
UNAME=$(shell uname -a | tr -c -d \[:alnum:\] | tr \[:lower:\] \[:upper:\] ) #upper uname-a
TMPDIR=/run/user/$(shell id -u)/
#MACROS
MACROS = -DDEBUGPRINT="if(FALSE)" -DDEBUG="if(FALSE)" -DCONSISTENCY_CHECKS="if(FALSE)" -DVERBOSE="if(FALSE)"  -DDEBUGCHECKS="if(FALSE)" 
MACROSDBG = -DCONSISTENCY_CHECKS="if(TRUE)"  -DDEBUGCHECKS="if(TRUE)" -DVERBOSE="if(TRUE)" -DDEBUG="if(TRUE)"
UNDEF := $(shell echo $(MACROSDBG) | tr " " "\n" | grep -oe '-D.*=' | tr -d "=" |  sed s/-D/-U/ )
CFLAGSDBG = -ggdb -O0 #-Og 	#TODO -O0 cause REDUCTION MISWRITE...FAULTY RESOULT (OR SERIAL ERROR)
#GATHER ALL DEBUG STUFF IN A VAR
DBG=$(CFLAGSDBG) $(UNDEF) -UAVG_TIMES_ITERATION -DAVG_TIMES_ITERATION=1 $(MACROSDBG)  
CONSTS = -DTMPDIR='"$(TMPDIR)"'  -DAVG_TIMES_ITERATION=40  -DDOUBLE_VECT_DIFF_EARLY_EXIT $(RUNTIME) #RUNTIME HAS TO BE AS LAST FOR CORRECT OVERRIDING

#bind to source original project
srcDir=$(realpath ../src)
srcInclDir=$(realpath ../src/include)
coreSrcs=$(wildcard $(srcDir)/Sp3MM*OMP*.c $(srcDir)/commons/*.c $(srcDir)/commons/ompi_dims_create/*.c  $(srcDir)/lib/*.c $(srcInclDir)/*.h )
testSrcs=Sp3MM_test.c  $(coreSrcs)

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
	#CFLAGS+= -fPIC #TODO NOT WORKING ON DELL FED BUILD.a
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

objs := $(shell  grep -Eo '.*\.o:\s' Makefile | grep -v -e '@' -e PHONY -e '##' | awk -F: '{print $1}' | tr '\n:' ' ' )


all: $(objs)

Sp3MM_getFlops.o: sp3MM_getFlops.c $(coreSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS)
Sp3MM_getFlops_DBG.o: sp3MM_getFlops.c $(coreSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG)

#.SECONDEXPANSION:
baseCompil=$(CFLAGS) -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$(testSrcs)) $(MACROS) $(CONSTS) 
$(info baseCompil:		$(baseCompil))

###QUIET TESTS - PERF GATHER
test_Sp3MM.o: $(testSrcs)							#dflts short
	$(CC) -o $@ $(baseCompil) -UAVG_TIMES_ITERATION -DAVG_TIMES_ITERATION=1
test_Sp3MM_Stats.o: $(testSrcs)						#dflts
	$(CC) -o $@ $(baseCompil)
test_Sp3MM_Stats_UB_DYN_ASSIGN.o: $(testSrcs)		#UB dynAssign
	$(CC) -o $@ $(baseCompil) -DSPARSIFY_PRE_PARTITIONING=F -DUB_IMPL_ONLY
test_Sp3MM_Stats_IDXFLAG_ARRAY.o: $(testSrcs)		#Char array instead of bitmaps limbs
	$(CC) -o $@ $(baseCompil) -DSPVECT_IDX_BITWISE=F
test_Sp3MM_Stats_IDXBITMAP_U64.o: $(testSrcs)		#ulong64 vars as bitmaps limbs
	$(CC) -o $@ $(baseCompil) -DLIMB_T=ulong
test_Sp3MM_Stats_IDXBITMAP_INT.o: $(testSrcs)		#uint vars as bitmaps limbs
	$(CC) -o $@ $(baseCompil) -DLIMB_T=uint
test_Sp3MM_Stats_SYMB_RB_NOCACHED.o: $(testSrcs)	#no cached insert in RBtrees
	$(CC) -o $@ $(baseCompil) -DRB_CACHED_INSERT=F


##DEBUG BUILDS
test_Sp3MM_DBG_UB_NN_TID_RESERVED.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) \
	$(CONSTS) $(MACROS) $(DBG) -DSPARSIFY_PRE_PARTITIONING=F
test_Sp3MM_DBG.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG)
##test_Sp3MM_FORTRAN_INTEGRATE_DBG.o: $(testSrcs)
##	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG) -DMOCK_FORTRAN_INDEXING

test_CBLAS_Sp3MM_OMP.elf: $(testSrcs) Sp3MM_CBLAS.c
	mkdir -p /tmp/oldBuilds
	mv *.o /tmp/oldBuilds
	#create objectfile not linked
	$(CC) -c $(CFLAGS) -I$(srcDir)/include/ -I$(CBLAS_INCLUDE) $(filter-out %.h %Generic.c,$^) \
	  $(CONSTS) $(MACROS) $(DBG) -DCBLAS_TESTS 
	#link it with static compiled cblas and source version library
	$(FF) -O2 -frecursive -o $@ *.o  $(CBLAS_SLIB) $(BLAS_SLIB) $(EXTRALIBS)
	rm -f *.o   #remove newly generated obj file for FF link
	mv /tmp/oldBuilds/* . 

test_utils.elf: 	$(srcDir)/commons/*.c $(srcDir)/lib/*.c $(srcInclDir)/*.h 
	$(CC) -o $@  $(CFLAGS) -I$(srcDir)/include/  $(filter-out %.h %Generic.c %Multi.c,$^) $(CONSTS) $(MACROS) $(DBG) \
	-DSPARSEUTILS_MAIN_TEST -O0

##ompGetICV.o:	$(srcDir)/commons/ompGetICV.c $(srcInclDir)/ompGetICV.h
##	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) -ggdb -D OMP_GET_ICV_MAIN 

testAll:
	$(shell bash ../scripts/testAllMatrixGroups.sh | tee - > log/testAll$GRID_ROWSXGRID_COLS )

sparseMatrixToImage.elf:	sparseMatrixToImage.c sparseMatrixToImage.h $(coreSrcs)
	$(CC) -o $@ $(CFLAGS) -I$(srcDir)/include -I. $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) \
	    -D MAIN_SPMAT_IMG 
sparseMatrixToImageTest.elf:	sparseMatrixToImage.c sparseMatrixToImage.h $(coreSrcs)
	$(CC) -o $@ $(CFLAGS) -I$(srcDir)/include -I. $(filter-out %.h,$^) $(CONSTS) $(MACROS) \
	    -D MAIN_SPMAT_IMG 	-DTEST  $(CFLAGSDBG)
clean:
	rm -f *.elf
	rm -f *.o

try:   
	echo $(TMPDIR) 
	echo $(objs) 
	echo $(UNDEF)
	@echo uname:        $(UNAME)
		
	

.PHONY: all clean testAll
