cat ../../data/matrixGroups.list | xargs -L 1 sh -c 'echo "## $0 ${@}";./test_CBLAS_SpGEMM_OMP.elf $0 ${@}'
