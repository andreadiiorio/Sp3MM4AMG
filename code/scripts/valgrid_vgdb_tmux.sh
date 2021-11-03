usage="<TRGT BIN TO DEBUG WITH VALGRID + GDB,[override args]>;\nexport SESSIONNAME"
if [ "$1" = "-h" ];then echo -e "$usage";exit 1;fi
if [ -z $1 ];then echo -e "$usage"; exit 1;fi
TRGTBIN=$(realpath "$1" )
ARGS=("r" "ac" "p" "ac_next")
#TODO OVERRIDE ARGS WITH $2,...
sesionName="$( echo $0--$(date -Iseconds) | tr ':./' '_' )"
if [[ $SESSIONNAME ]]; then sessionName=$SESSIONNAME; fi
VALGRIND_SPAWN_TIME=1
VGDB="/usr/libexec/valgrind/../../bin/vgdb" #PATH FROM VALGRIND..
VALGRINDCMD="valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --vgdb=yes --vgdb-error=0 $TRGTBIN ${ARGS[@]}  2>&1 | less"
GDBCMD='gdb -ex "target remote | vgdb";kill -9 $(pidof valgrind)'";tmux kill-session $sessionName"


unset TMUX  #allow tmux nesting
##TODO, gdb doesn't work ... 
tmux new-session   -d -s $sesionName >/dev/null && tmux splitw -h -t $sesionName:0.0 && \
tmux send-keys -t $sesionName:0.0 "bash -c '$VALGRINDCMD'" Enter && sleep $VALGRIND_SPAWN_TIME && \
tmux send-keys -t $sesionName:0.1 "bash -c '$GDBCMD'" Enter && \
tmux a -t $sesionName 
