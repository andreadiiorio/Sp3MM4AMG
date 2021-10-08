objs  = SpGEMM_OMP.o SpGEMM_CUDA.o test.o
srcs  = $(find -name "*.c" )
srcs += $(find -name "*.h" )

all: $(objs)
.PHONY: all

SpGEMM_OMP.o: $(srcs)
	$(MAKE) -C src $@
	ln -fs src/$@ .
test_SpGEMM_OMP.o:	$(srcs)
	$(MAKE) -C test $@
	ln -fs test/$@ .
clean:
	$(MAKE) -C src $@
	rm -i *.o 
