#Copyright Andrea Di Iorio 2022
#This file is part of Sp3MM_for_AlgebraicMultiGrid
#Sp3MM_for_AlgebraicMultiGrid is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#Sp3MM_for_AlgebraicMultiGrid is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with Sp3MM_for_AlgebraicMultiGrid.  If not, see <http://www.gnu.org/licenses/>.

CC=gcc
FF=gfortran
#libs
#extra to reduce useless warnings
CWALL =-Wall -Wextra
CWALL+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-switch
CWALL+=-Wno-unused-function -Wno-unused-variable       #TODO CLEAN USELESS FUNCTIONS / VARIABLES
CWALL+=-Wno-unused-label -Wfatal-errors
CFLAGS=-O2 $(RUNTIME) $(CWALL)
CFLAGS+=-lm -fopenmp

#SYSTEM CONFIGURATION
UNAME=$(shell uname -a | tr -c -d \[:alnum:\] | tr \[:lower:\] \[:upper:\] ) #upper uname-a
TMPDIR=/run/user/$(shell id -u)/
#MACROS
CONSTS = -DTMPDIR='"$(TMPDIR)"'  -DAVG_TIMES_ITERATION=25  -DDOUBLE_VECT_DIFF_EARLY_EXIT 
MACROS = -DDEBUGPRINT="if(FALSE)" -DDEBUG="if(FALSE)" -DCONSISTENCY_CHECKS="if(FALSE)" -DVERBOSE="if(FALSE)"  -DDEBUGCHECKS="if(FALSE)" 
MACROSDBG = -DCONSISTENCY_CHECKS="if(TRUE)"  -DDEBUGCHECKS="if(TRUE)" -DVERBOSE="if(TRUE)" -DDEBUG="if(TRUE)"
UNDEF := $(shell echo $(MACROSDBG) | tr " " "\n" | grep -oe '-D.*=' | tr -d "=" |  sed s/-D/-U/ )
CFLAGSDBG = -ggdb -O0 #-Og 	#TODO -O0 cause REDUCTION MISWRITE...FAULTY RESOULT (OR SERIAL ERROR)
#GATHER ALL DEBUG STUFF IN A VAR
DBG=$(CFLAGSDBG) $(UNDEF) -UAVG_TIMES_ITERATION -DAVG_TIMES_ITERATION=1 $(MACROSDBG)  

#bind to source original project
srcDir=$(realpath ../src)
srcInclDir=$(realpath ../src/include)
coreSrcs:=$(srcDir)/Sp3MM*OMP*.c $(srcDir)/commons/*.c  $(srcDir)/lib/*.c $(srcInclDir)/*.h 
testSrcs:=Sp3MM_test.c  $(coreSrcs)

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

objs := $(shell  grep -Eo '.*\.o:\s' Makefile | grep -v -e '@' -e PHONY | awk -F: '{print $1}' | tr '\n:' ' ' )


all: $(objs)
#FAST TESTS - PERFGATHER
test_Sp3MM.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS) -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) \
	$(MACROS) $(CONSTS) -UAVG_TIMES_ITERATION -DAVG_TIMES_ITERATION=1
test_Sp3MM_Stats.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS) -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(MACROS) $(CONSTS)
test_Sp3MM_Stats_UB_NN_TID_RESERVED.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS) -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) \
	$(MACROS) $(CONSTS) -DSPARSIFY_PRE_PARTITIONING=F

##DEBUG BUILDS
test_Sp3MM_DBG_UB_NN_TID_RESERVED.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) \
	$(CONSTS) $(MACROS) $(DBG) -DSPARSIFY_PRE_PARTITIONING=F
test_Sp3MM_DBG.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG)
test_Sp3MM_FORTRAN_INTEGRATE_DBG.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS)  -I$(srcDir)/include/  $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG) -DMOCK_FORTRAN_INDEXING

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

ompGetICV.o:	$(srcDir)/commons/ompGetICV.c $(srcInclDir)/ompGetICV.h
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) -ggdb -D OMP_GET_ICV_MAIN 

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
