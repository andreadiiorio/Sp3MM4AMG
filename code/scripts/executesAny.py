"""
pandas based analisis over Sp3MM parsed logs 
================================================================================
export:     ...outMetrics...
usage:      <log0,label0,log1,label1,....>
"""

from re             import finditer
from sys            import argv,stderr
from os             import environ as env
from parseLogSp3MM  import parseLog

from pandas import  *

if __name__ == "__main__":
    l = len(argv)
    if l < 3 or l-1%2 == 1 or "-h" in argv[1]:    print(__doc__,file=stderr);exit(1)
    logsDFs = list()
    for x in range(int(l/2)):
        logFname,logLabel = argv[1+ x*2],argv[1+ x*2+1]
        with open(logFname) as f:   logExecutesTuples = parseLog(f.read())
        logEntriesDF                = DataFrame(logExecutesTuples)
        logEntriesDF["logLabel"]    = logLabel 
        logsDFs.append(logEntriesDF)
    a = logEntriesDF.describe()
    print("end")

