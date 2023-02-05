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

srcDir=$(realpath ../src)
srcInclDir=$(realpath ../src/include)
coreSrcs=$(wildcard $(srcDir)/Sp3MM*OMP*.c $(srcDir)/commons/*.c $(srcDir)/commons/ompi_dims_create/*.c  $(srcDir)/lib/*.c $(srcInclDir)/*.h )
testSrcs=Sp3MM_test.c  $(coreSrcs)

CWALL =-Wall -Wextra
#extra to reduce useless warnings
CWALL+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-switch
CWALL+=-Wno-unused-function -Wno-unused-variable       #TODO CLEAN USELESS FUNCTIONS / VARIABLES
CWALL+=-Wno-unused-label -Wfatal-errors

CINCL   = -I$(srcDir)/include/ 
CFLAGS  = -O2 $(CWALL) $(CINCL) -fopenmp
LDFLAGS = -lm

#SYSTEM CONFIGURATION
TMPDIR=/run/user/$(shell id -u)/
#MACROS
MACROS = -DDEBUGPRINT="if(FALSE)" -DDEBUG="if(FALSE)" -DCONSISTENCY_CHECKS="if(FALSE)" -DVERBOSE="if(FALSE)"  -DDEBUGCHECKS="if(FALSE)" 
MACROSDBG = -DCONSISTENCY_CHECKS="if(TRUE)"  -DDEBUGCHECKS="if(TRUE)" -DVERBOSE="if(TRUE)" -DDEBUG="if(TRUE)"
UNDEF := $(shell echo $(MACROSDBG) | tr " " "\n" | grep -oe '-D.*=' | tr -d "=" |  sed s/-D/-U/ )
CFLAGSDBG = -ggdb -O0 #-Og 	#TODO -O0 cause REDUCTION MISWRITE...FAULTY RESOULT (OR SERIAL ERROR)
#GATHER ALL DEBUG STUFF IN A VAR
DBG=$(CFLAGSDBG) $(UNDEF) -UAVG_TIMES_ITERATION -DAVG_TIMES_ITERATION=1 $(MACROSDBG)  
CONSTS = -DTMPDIR='"$(TMPDIR)"'  -DAVG_TIMES_ITERATION=40  -DDOUBLE_VECT_DIFF_EARLY_EXIT $(RUNTIME) #RUNTIME HAS TO BE AS LAST FOR CORRECT OVERRIDING

objs := $(shell  grep -Eo '^.*\.o:\s' Makefile | grep -v -e '@' -e PHONY -e '\#\#' | awk -F: '{print $1}' | tr '\n:' ' ' )
#objs := Sp3MM_getFlops.o Sp3MM_getFlops_DBG.o test_Sp3MM.o test_Sp3MM_Stats.o test_Sp3MM_Stats_UB_DYN_ASSIGN.o test_Sp3MM_Stats_IDXFLAG_ARRAY.o test_Sp3MM_Stats_IDXBITMAP_U64.o test_Sp3MM_Stats_IDXBITMAP_INT.o test_Sp3MM_Stats_SYMB_RB_NOCACHED.o test_Sp3MM_DBG_UB_NN_TID_RESERVED.o test_Sp3MM_DBG.o

all: $(objs)

Sp3MM_getFlops.o: sp3MM_getFlops.c $(coreSrcs)
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(LDFLAGS)
Sp3MM_getFlops_DBG.o: sp3MM_getFlops.c $(coreSrcs)
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG) $(LDFLAGS)

#.SECONDEXPANSION:
baseCompil=$(CFLAGS) $(filter-out %.h %Generic.c,$(testSrcs)) $(MACROS) $(CONSTS) $(LDFLAGS)
$(info baseCompil:		$(baseCompil))

###QUIET TESTS - PERF GATHER
test_Sp3MM.o: $(testSrcs)
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
	$(CC) -o $@ $(CFLAGS)   $(filter-out %.h %Generic.c,$^) \
	$(CONSTS) $(MACROS) $(DBG) -DSPARSIFY_PRE_PARTITIONING=F $(LDFLAGS)

test_Sp3MM_DBG.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG) $(LDFLAGS)

test_Sp3MM_FORTRAN_INTEGRATE_DBG.o: $(testSrcs)
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) $(CONSTS) $(MACROS) $(DBG) -DMOCK_FORTRAN_INDEXING $(LDFLAGS)

ompGetICV.o:	$(srcDir)/commons/ompGetICV.c $(srcInclDir)/ompGetICV.h
	$(CC) -o $@ $(CFLAGS) $(filter-out %.h %Generic.c,$^) -ggdb -D OMP_GET_ICV_MAIN  $(LDFLAGS)

testAll:
	bash testAllMatrixGroups.sh | tee test.log

clean:
	rm -i *.elf
	rm -i *.o

try:   
	echo $(objs) 
	echo $(TMPDIR) 
	echo $(UNDEF)
	@echo uname:        $(UNAME)
		
	

.PHONY: all clean testAll
