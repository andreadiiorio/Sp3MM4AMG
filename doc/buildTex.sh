pdflatex="~/DATA/SW/latex/2021/bin/x86_64-linux/pdflatex -jobname=out " #-output-directory=build " #-halt-on-error"
bibtex="~/DATA/SW/latex/2021/bin/x86_64-linux/bibtex"
#rm out* *aux
eval $pdflatex  $1
if [[ $? != 0 ]];then exit 1;fi
if [[ -z $NO_REFERENCES ]];then		#rebuilds to make reference available
	eval $bibtex out; eval $pdflatex  $1&& eval $pdflatex  $1
fi
if [[ $? ]];then evince "out.pdf";fi
