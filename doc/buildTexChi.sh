trgtCh="Chapters/ChIntro"
trgtChOutName="ch"
if [[ $1 ]];then				trgtCh="$1";fi
if [[ $TRGTCHOUTNAME ]];then	trgtChOutName="$TRGTCHOUTNAME";fi

mkdir -p build
pdflatex="~/DATA/SW/latex/2021/bin/x86_64-linux/pdflatex -output-directory=build " #-halt-on-error"
bibtex="~/DATA/SW/latex/2021/bin/x86_64-linux/bibtex"
trgtChLimit=" -jobname=$trgtChOutName \"\includeonly{$trgtCh}\input{main.tex}\""
################################################################################
eval $pdflatex "$trgtChLimit" main.tex
if [[ $? != 0 ]];then exit 1;fi
if [[ -z $NO_REFERENCES ]];then		#rebuilds to make reference available
	eval $bibtex $trgtChOutName; 
	eval $pdflatex "$trgtChLimit" main.tex && \
	eval $pdflatex "$trgtChLimit" main.tex
fi
if [[ $? ]];then evince "build/$trgtChOutName".pdf;fi
