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
