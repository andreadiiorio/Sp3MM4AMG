trgtCh="Chapters/ChIntro"
trgtChOutName="ChIntro"
if [[ $TRGTCH ]];then			trgtCh="$TRGTCH";fi
if [[ $TRGTCHOUTNAME ]];then	trgtChOutName="$TRGTCHOUTNAME";fi

pdflatex="~/DATA/SW/latex/2021/bin/x86_64-linux/pdflatex"
bibtex="~/DATA/SW/latex/2021/bin/x86_64-linux/bibtex"
trgtChLimit=" -jobname=$trgtChOutName \"\includeonly{$trgtCh}\input{main.tex}\""
################################################################################
eval $pdflatex "$trgtChLimit" main.tex && \
eval $bibtex $trgtChOutName; \
eval $pdflatex "$trgtChLimit" main.tex && \
eval $pdflatex "$trgtChLimit" main.tex &&\
 evince "$trgtChOutName".pdf
