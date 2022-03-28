#build kind of index of a series of latex files, exploing sub in sections for tabbing titles
help="[tex files list || find from here for \"*.tex\"*]"
if [[ "$1" = "-h" ]];then echo $help 1>&2;fi
args=( $( find -name "*.tex" ) )
if [[ "$1" ]];then args=( "$1" );fi
echo -e "index regxPlace build over files:\t ${args[@]}\n\n"
grep  "section{.*}" --no-filename ${args[@]} | tr -d '\\' |sed 's/section//' | sed 's/sub/\t/g'
