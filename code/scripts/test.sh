#-DDECREASE_THREAD_NUM=F
mkdir log log/schedDyn log/schedStatic 
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats.o						> log/schedDyn/test_Sp3MM_Stats
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_UB_DYN_ASSIGN.o		> log/schedDyn/test_Sp3MM_Stats_UB_DYN_ASSIGN.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_IDXBITMAP_U64.o		> log/schedDyn/test_Sp3MM_Stats_IDXBITMAP_U64.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_IDXBITMAP_INT.o		> log/schedDyn/test_Sp3MM_Stats_IDXBITMAP_INT.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_SYMB_RB_NOCACHED.o		> log/schedDyn/test_Sp3MM_Stats_SYMB_RB_NOCACHED.log
#bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_IDXFLAG_ARRAY.o		> log/schedDyn/ttttt
export OMP_SCHEDULE="nonmonotonic:static"
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats.o						> log/schedStatic/test_Sp3MM_Stats
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_UB_DYN_ASSIGN.o		> log/schedStatic/test_Sp3MM_Stats_UB_DYN_ASSIGN.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_IDXBITMAP_U64.o		> log/schedStatic/test_Sp3MM_Stats_IDXBITMAP_U64.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_IDXBITMAP_INT.o		> log/schedStatic/test_Sp3MM_Stats_IDXBITMAP_INT.log
bash testAllMatrixGroups.sh ../test/test_Sp3MM_Stats_SYMB_RB_NOCACHED.o		> log/schedStatic/test_Sp3MM_Stats_SYMB_RB_NOCACHED.log