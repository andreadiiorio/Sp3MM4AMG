Sparse Triple Matrix Matrix - Multiplication For Algebraic Multigrid
====================================================================

Master Degree experimental thesis about implementations of 
Triple Sparse Matrix Matrix Multiplication in C with OpenMP.

This operation is suitable to support the Matrices hierarchy construction in the
setup of the Algebraic Multi Grid methods.

I've focused on the (ordered) CSR representations of the sparse matrices

# Quick Overview of main features
- 1D and 2D data partitioning scheme for threads in openMP
- both upper bound and accurate symbolic step as initial phase
- direct computation of triple product by forwarding row-by-row, gustvason based approach, to the third matrix
- single function multi implementation via multiple included sources
  enabling easy support for 1-based version of each needed function for integration in FORTRAN projects
- several tuning parameters and configuration macros for different implementations options

# Organization and use
.
|-- code
|   |-- src         implementations source file
|   |   |-- commons
|   |   |-- include
|   |   `-- lib
|   `-- test
|       `-- Sp3MM_test.c    application to wrap invocation of several
|       |                   implementation and check the result
|       `-- Makefile        main makefile 

|-- data            Compressed 4-tuple of sparse matrices to perform the
|                   multiplication and check the results
`-- doc             Thesis latex sources

Running a test is possible invoking the compiled binary with the 3 input matrices
and optionally the result matrix (used to check correctness, if not given a
reference implementation will be used)
e.g. using the sample matrices in the test folder
`./test_Sp3MM.o    r ac p ac_next`

Building is as easy as invoking make -j$(nproc)

NB the source code of this repository is released with the BSD3-CLAUSE license
but only the red-black tree implementations are with the original linux GPL license.
In the branch bsd3-only it's removed the GPL licensed code

# Implementations
## Upper Bound
Over-allocation of each thread computed matrix part and smart re-assembling

Less cost in the setup but it's needed a later merge of the threads' outputs
to have a well formed CSR matrix

Extra space can be assigned on the fly to each thread if defined macros
-DSPARSIFY_PRE_PARTITIONING=F -DUB_IMPL_ONLY
`\__atomic_fetch_add`-like will be used to have every thread to book safely a mem
array for the data he's to (temporally) save

## Symbolic - Numeric
Computation of a sparse matrix - matrix product can be divided in a 
symbolic phase where only the non-zero number and pattern is counted.
This way, each matrix portion computed in parallel is exactly pre-allocable
and safely written,
so paying an extra setup cost we save a merging phase at the end, compared with
the UpperBound version.

### Linux RB-Tree
### IDX BITMAP

## Direct Triple Multiplication

# Compilation Macros
- -DLIMB_T=<ulong|uint>     change bitmap limbs length in idx bitmap
                            implementation and sparsification operation

- -DSPARSIFY_PRE_PARTITIONING=F -DUB_IMPL_ONLY
                            dyn assign of pre-allocated upperbounded matrix to
                            threads via an atomic operation
- -DRB_CACHED_INSERT=F      don't use linux rbtree leftmost cached version

- -DMOCK_FORTRAN_INDEXING   manually pre shift the parsed matrix and 
                            use 1based implementations' functions
                            to try FORTRAN integration