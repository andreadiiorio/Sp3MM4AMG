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
CFLAGS= -O2 # -std=c99 #TODO reallocarray
#libs
CFLAGS+=-lm -fopenmp
CFLAGSDBG=$(CFLAGS) -ggdb -Og

#TODO extra to reduce useless warnings
CWALL+=-Wall -Wextra -Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable
CWALL+=-Wno-unused-label -Wno-switch
CWALL+=-Wno-unused-function -Wno-unused-variable -Werror=incompatible-pointer-types -Werror=misleading-indentation
CFLAGS+=$(RUNTIME) $(CWALL) -Wfatal-errors 
##TODO OMP CONFIG
#OMP_CANCELLATION=true
#export OMP_CANCELLATION

#SYSTEM CONFIGURATION
UNAME=$(shell uname -a | tr -c -d \[:alnum:\] | tr \[:lower:\] \[:upper:\] ) #upper uname-a
TMPDIR=/run/user/$(shell id -u)/

#MACROS
CONSTS = -DTMPDIR='"$(TMPDIR)"'  -DAVG_TIMES_ITERATION=25  -DDOUBLE_VECT_DIFF_EARLY_EXIT 
MACROS = -DDEBUGPRINT="if(FALSE)" -DDEBUG="if(FALSE)" -DCONSISTENCY_CHECKS="if(FALSE)" -DVERBOSE="if(FALSE)"  -DDEBUGCHECKS="if(FALSE)" 
MACROSDBG = -DCONSISTENCY_CHECKS="if(TRUE)"  -DDEBUGCHECKS="if(TRUE)" -DVERBOSE="if(TRUE)" -DDEBUG="if(TRUE)"
UNDEF := $(shell echo $(MACROSDBG) | tr " " "\n" | grep -oe '-D.*=' | tr -d "=" |  sed s/-D/-U/ )
DBG= $(UNDEF) $(MACROSDBG)  

objs = SpMM_OMP.o #SpMM_CUDA.o
all: $(objs)
.PHONY: all

coreSrcs = Sp*MM*OMP*.c commons/*.c lib/*.c include/*.h
Sp3MM_OMP.o: $(coreSrcs) main_OMP.c 
	$(CC) -o $@ $(CFLAGS) $(CONSTS) $(MACROS) -I include/ \
      $(filter-out %.h %Generic.c, $^)
Sp3MM_OMP.a: $(coreSrcs)
	$(CC) -c $(CFLAGS) $(CONSTS) $(MACROS) -I include/ \
      $(filter-out %.h %Generic.c, $^)
	#TODO ar rcs $@ *.o

#DEBUG VERSION
Sp3MM_OMP_DBG.a: $(coreSrcs)
	$(CC) -c $(CFLAGSDBG) $(CONSTS) $(MACROS) $(DBG) -I include/ \
      $(filter-out %.h %Generic.c, $^)
	ar rcs $@ *.o
clean:
	#rm *.o #-i 
	#rm *.a #-i
	mv *.o /tmp	
	mv *.a /tmp	

try: $(filter-out %_Generic.c,main_OMP.c SpMM*OMP_Multi.c commons/*.c lib/*.c include/*.h)
	echo $(filter-out %Generic.c %Generic.h,$^)

#MAKEFILE TESTS FEATURES
headersAll = $(find -name "*.h")
try2: $(headersAll) main_OMP.c include/*
	@echo ciao $(filter-out %.c, $^)
	python3 -c 'from os import environ as env;print(env.get("OMP_CANCELLATION"))'
