CC=gcc
CFLAGS= -ggdb -O2 #-std=c99 #TODO reallocarray
#libs
CFLAGS+=-lm -fopenmp

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

CONSTS = -DTMPDIR='"$(TMPDIR)"'     #TODO PUT IN /raccolta
MACROS = -DDEBUGPRINT="if(FALSE)" -DCONSISTENCY_CHECKS="if(TRUE)"
MACROS+= -DVERBOSE="if(FALSE)"

UNDEF := $(shell echo $(MACROSDBG) | tr " " "\n" | grep -oe '-D.*=' | tr -d "=" |  sed s/-D/-U/ )


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
