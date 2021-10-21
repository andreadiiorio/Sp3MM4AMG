"""
parse test_CBLAS_SpGEMM_OMP.elf output log over a large set of matrixes groups into a CSV on stdout
source matrixes lines starts with ##, computing and configuration lines start with @
expected prefixed lines in this order + template
##sources ... #source
@sizes and config
@compute ... func:X elapsed:XX internalTime:X
TEMPLATE:
## /home/andreadiiorio/data/tesi/VanekBrezina/Large/Smoothed/dump_lev_d_p0_l003_r.mtx /home/andreadiiorio/data/tesi/VanekBrezina/Large/Smoothed/dump_lev_d_p0_l002_ac.mtx /home/andreadiiorio/data/tesi/VanekBrezina/Large/Smoothed/dump_lev_d_p0_l003_p.mtx /home/andreadiiorio/data/tesi/VanekBrezina/Large/Smoothed/dump_lev_d_p0_l003_ac.mtx # ./VanekBrezina/Large/Smoothed 2
preparing time: 1.226807e+00    @COARSENING AC: 64560x64560 ---> 1813x1813      conf grid: 40x8,        NNZ:326894-1903196-326894                            
@computing Sp3GEMM as pair of SpGEMM with func:0 at:0x403360    elapsed 4.291327e-02 - flops 3.883883e+10       internalTime: 4.169533e-03                   
@computing Sp3GEMM as pair of SpGEMM with func:1 at:0x402990    elapsed 4.628078e-02 - flops 3.601283e+10       internalTime: 6.217946e-03                   
@computing Sp3GEMM as pair of SpGEMM with func:2 at:0x403890    elapsed 4.590042e-02 - flops 3.631125e+10       internalTime: 4.198987e-03                   
all spgemmFuncs passed the test

usage <logFile>
"""
from collections import namedtuple
from re import finditer
from sys import argv

getReGroups=lambda pattern,string:\
    finditer(pattern,string).__next__().groups()
getReMatch=lambda pattern,string:\
    finditer(pattern,string).__next__().groups()[0]
def getReMatchFull(pattern,string):
    #return first occurence of @pattern in @string or ""
    try:
        a=finditer(pattern,string).__next__()
        x,y=a.span()
        out=a.string[x:y]
    except: out=""
    return out

GRID_PATTERN="[0-9]+x[0-9]+"
FP_PATTERN="[-+]?\d+\.?\d+e[-+]\d+"
parseSizes=lambda s:  s #[int(x) for x in s.split("x")] #TODO not good CSV PARSED



FIELDS = "source,funcN,time,internalTime,preparingTime,srcSize,dstSize,NNZ_R,NNZ_AC,NNZ_P,gridSize"
Execution = namedtuple("Execution",FIELDS)

if "-h" in argv[1] or len(argv)<2:  print(__doc__);exit(1)

executionTimes = list() #Execution tups
with open(argv[1]) as f:    log=f.read()
linesGroup = [ g.split("\n") for g in log.split("##")]

for i,g in enumerate(linesGroup):
    if len(g) < 3:  print("not complete group",i,g);continue
    header   = g[0]
    configSiz= g[1]
    computes = list(filter(lambda l:"@" in l,g[2:]))
    src      = header[header.rfind("#"):].replace(" ","_")
    srcSize  = parseSizes(getReMatch("COARSENING AC:\s*("+GRID_PATTERN+")",configSiz))
    dstSize  = parseSizes(getReMatch("-->\s*("+GRID_PATTERN+")",configSiz))
    gridSize = parseSizes(getReMatch("grid:\s*("+GRID_PATTERN+")",configSiz))
    nnz_racp=getReMatch("NNZ:\s*(\d+-\d+-\d+)",configSiz)
    nnz_r,nnz_ac,nnz_p=[int(x) for x in nnz_racp.split("-")]
    preparingTime = float(getReMatch("preparing time:\s*("+FP_PATTERN+")",configSiz))
    for l in computes:
        funcN   = int(getReMatch("func:\s*(\d)",l))
        elapsed = float(getReMatch("elapsed\s*("+FP_PATTERN+")",l))
        internalTime = float(getReMatch("internalTime:\s*("+FP_PATTERN+")",l))
        executionTimes.append(Execution(src,funcN,elapsed,internalTime,preparingTime,srcSize,dstSize,nnz_r,nnz_ac,nnz_p,gridSize))

print(FIELDS)
for e in executionTimes:    
    for f in e: print(f,end=", ")
    print("")
print("\n")
