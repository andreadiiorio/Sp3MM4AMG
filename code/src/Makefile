CC=gcc
CFLAGS=-Wall -Wextra -ggdb -O2 #-std=c99 #TODO reallocarray
#libs
CFLAGS+=-lm

#TODO extra to reduce useless warnings
CFLAGS+=-Wno-pointer-sign -Wno-unused-parameter -Wno-unused-but-set-variable
CFLAGS+=-Wno-unused-label

MACROS = -DDEBUGPRINT="if(FALSE)" -DCONSISTENCY_CHECKS="if(TRUE)"
MACROS+= -DVERBOSE="if(FALSE)"
#OMP CONFIG
OMP_CANCELLATION=true
export OMP_CANCELLATION
#TODO ALTERNATIVAMENTE CON error severity fatal at execution

objs = SpGEMM_OMP.o SpGEMM_CUDA.o test.o

all: $(objs)
.PHONY: all

SpGEMM_OMP.o: main_OMP.c SpGEMM_OMP.c commons/*.c lib/parser.c lib/mmio.c include/*.h
	$(CC) -fopenmp $(CFLAGS) $(MACROS) -I include/ $(filter-out %.h,$^) -o $@ # -O0
clean:
	rm -i *.o 


#MAKEFILE TESTS FEATURES
headersAll = $(find -name "*.h")
try: $(headersAll) main_OMP.c include/*
	@echo ciao $(filter-out %.c, $^)
	python3 -c 'from os import environ as env;print(env.get("OMP_CANCELLATION"))'
