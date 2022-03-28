#Copyright Andrea Di Iorio 2022
#This file is part of Sp3MM_for_AlgebraicMultiGrid
#Sp3MM_for_AlgebraicMultiGrid is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#Sp3MM_for_AlgebraicMultiGrid is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with Sp3MM_for_AlgebraicMultiGrid.  If not, see <http://www.gnu.org/licenses/>.
"""
parse test_CBLAS_SpGEMM_OMP.elf output log over a large set of matrixes groups into a CSV on stdout
source matrixes lines starts with ##, computing and configuration lines start with @
expected prefixed lines in this order + template
##sources ... #source
@sizes and config
@compute ... func:X timeAvg:XXX timeVar:XXX timeInternalAvg:XXX timeInternalVar:XXX
TEMPLATE:
## ../../data/Matching/Small/Unsmoothed/dump_lev_d_p0_l003_r.mtx ../../data/Matching/Small/Unsmoothed/dump_lev_d_p0_l002_ac.mtx ../../data/Matching/Small/Unsmoothed/dump_lev_d_p0_l003_p.mtx 
../../data/Matching/Small/Unsmoothed/dump_lev_d_p0_l003_ac.mtx # ../../data/Matching/Small/Unsmoothed 2                                                                                       
@COARSENING AC: 343x343 ---> 70x70      conf grid: 8x8, NNZ:343-2261-343                                                                                                                      
@computing Sp3GEMM as pair of SpGEMM with func:0 at:0x403610    R:70x343 AC:343x343 P:343x70 CSR sp.Mat timeAvg:4.546160e-02 timeVar:4.173552e-09       timeInternalAvg:4.536528e-02 timeInternalVar:3.463142e-09                                                                                                                                                                           
@computing Sp3GEMM as pair of SpGEMM with func:1 at:0x403930    R:70x343 AC:343x343 P:343x70 CSR sp.Mat timeAvg:4.508296e-02 timeVar:1.414607e-05       timeInternalAvg:4.505302e-02 timeInternalVar:1.414472e-05                                                                                                                                                                           
@computing Sp3GEMM as pair of SpGEMM with func:2 at:0x403250    R:70x343 AC:343x343 P:343x70 CSR sp.Mat timeAvg:1.607386e-02 timeVar:3.895463e-04       timeInternalAvg:1.599381e-02 timeInternalVar:3.894364e-04                                                                                                                                                                           
@computing Sp3GEMM as pair of SpGEMM with func:3 at:0x403c80    R:70x343 AC:343x343 P:343x70 CSR sp.Mat timeAvg:4.049085e-04 timeVar:8.014725e-10       timeInternalAvg:1.945628e-04 timeInternalVar:3.897424e-10                                                                                                                                                                           
all spgemmFuncs passed the test

usage <logFile>
"""
from collections import namedtuple
from re import finditer
from sys import argv,stderr

FIELDS = "source,funcN,timeAvg,timeVar,internalTimeAvg,internalTimeVar,srcSize,dstSize,NNZ_R,NNZ_AC,NNZ_P,gridSize"
Execution = namedtuple("Execution",FIELDS)

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

def parseHeader(header):
    return header[header.rfind("#"):].strip().replace(" ","_") #src
def parseConfigSize(configSiz):
    srcSize  = parseSizes(getReMatch("COARSENING AC:\s*("+GRID_PATTERN+")",configSiz))
    dstSize  = parseSizes(getReMatch("-->\s*("+GRID_PATTERN+")",configSiz))
    gridSize = parseSizes(getReMatch("grid:\s*("+GRID_PATTERN+")",configSiz))
    nnz_racp=getReMatch("NNZ:\s*(\d+-\d+-\d+)",configSiz)
    nnz_r,nnz_ac,nnz_p=[int(x) for x in nnz_racp.split("-")]
    return srcSize,dstSize,gridSize,nnz_r,nnz_ac,nnz_p

def parseComputeTimes(l):
    funcN   = int(getReMatch("func:\s*(\d)",l))
    timeAvg = float(getReMatch("timeAvg:\s*("+FP_PATTERN+")",l))
    timeVar = float(getReMatch("timeVar:\s*("+FP_PATTERN+")",l))
    timeInternalAvg = float(getReMatch("timeInternalAvg:\s*("+FP_PATTERN+")",l))
    timeInternalVar = float(getReMatch("timeInternalVar:\s*("+FP_PATTERN+")",l))
    return funcN,timeAvg,timeVar,timeInternalAvg,timeInternalVar
if __name__ == "__main__":
    if "-h" in argv[1] or len(argv)<2:  print(__doc__);exit(1)
    
    executionTimes = list() #Execution tups
    with open(argv[1]) as f:    log=f.read()
    linesGroup = [ g.split("\n") for g in log.split("##")]
    
    for i,g in enumerate(linesGroup):
        if len(g) < 3:  print("not complete group",i,g,file=stderr);continue
        #splitting log parts of computation
        header   = g[0]
        configSiz= g[1]
        computes = list(filter(lambda l:"@" in l,g[2:]))
        #parsing
        src = parseHeader(header)
        srcSize,dstSize,gridSize,nnz_r,nnz_ac,nnz_p = parseConfigSize(configSiz)
        #preparingTime = float(getReMatch("preparing time:\s*("+FP_PATTERN+")",configSiz))
        for l in computes:
            funcN,timeAvg,timeVar,timeInternalAvg,timeInternalVar = parseComputeTimes(l)
            executionTimes.append(Execution(src,funcN,timeAvg,timeVar,timeInternalAvg,timeInternalVar,srcSize,dstSize,nnz_r,nnz_ac,nnz_p,gridSize))
    
    print(FIELDS)
    for e in executionTimes:    
        for f in e: print(f,end=", ")
        print("")
    print("\n")
