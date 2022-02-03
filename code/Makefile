objs  = SpMM_OMP.o SpMM_CUDA.o test.o
srcs  = $(find -name "*.c" )
srcs += $(find -name "*.h" )

all: $(objs)
.PHONY: all

SpMM_OMP.o: $(srcs)
	$(MAKE) -C src $@
	ln -fs src/$@ .
test_SpMM_OMP.o:	$(srcs)
	$(MAKE) -C test $@
	ln -fs test/$@ .
clean:
	$(MAKE) -C src  $@
	$(MAKE) -C test $@
	rm -i *.o 
