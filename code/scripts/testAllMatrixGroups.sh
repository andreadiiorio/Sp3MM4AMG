set -e -o pipefail
trgtbin="test_Sp3MM.o"
if [ $1 ];then trgtbin="$1";fi
export TRGTBIN="$trgtbin"
#cat matrixGroups.list | xargs -L 1 sh -ec 'echo "## $0 ${@}"; $(realpath $TRGTBIN ) $0 ${@}' #TODO no error termination
while read line; do
	matrixes=("$line")
	#echo "## ${matrixes[@]}"
	$(realpath $TRGTBIN ) ${matrixes[@]}
done < matrixGroups.list
