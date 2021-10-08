#NOTE SU BLAS E SCI SW NOTATIONS

#HP
Matrici N x M
...


##LeadingDimension
In general leading dimension __LD = number of elements in major dimension__
Also it isequal to the distance in elements between two neighbor elements in a line of minor dimension
caso ColumnMajorOrdering: LD = distanza tra 2 colonne consecutive = N
caso RowMajorOrdering:    LD = distanza tra 2 righe   consecutive = M


---
IMPLEMENTATIONS
---

versione su netlib dovrebbe essere una semplice reference implementation -> nn ottimizzata!
versioni ottimizzate
- ATLAS -> auto build con ottimizzazioni per arch http://www.netlib.org/blas/blas-3.10.0.tgz
- OpenBLAS -> alternative competitive a blas dei vendor... https://www.openblas.net/

VENDOR IMPLEMENATION
- intel -> https://software.intel.com/en-us/intel-mkl 


#CORE SW NAMING
#LAPACK
The distribution contains 
- the Fortran source for LAPACK, 
- its testing programs.  
- the Fortran reference implementation of the Basic Linear Algebra Subprograms (the Level 1, 2, and 3 BLAS) needed by LAPACK.  
_this code is intended for use only if there is no other
implementation of the BLAS already available on your machine; the efficiency of LAPACK depends very much on the efficiency of the BLAS_  
- CBLAS, a C interface to the BLAS, 
- LAPACKE, a C interface to LAPACK.

