set -e
trgtbin="test_SpGEMM_OMP.elf"
if [ $1 ];then trgtbin="$1";fi
export TRGTBIN="$trgtbin"
cat matrixGroups.list | xargs -L 1 sh -ec 'echo "## $0 ${@}"; $(realpath $TRGTBIN ) $0 ${@}'
