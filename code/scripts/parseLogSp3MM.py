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
source matrixes lines starts with #, computing and configuration lines start with @
expected prefixed lines in this order + template
##sources ... #source
@sizes and config
@compute ... func:X timeAvg:XXX timeVar:XXX timeInternalAvg:XXX timeInternalVar:XXX
TEMPLATE:
## r ac p 
../../data/Matching/Small/Unsmoothed/dump_lev_d_p0_l003_ac.mtx # ../../data/Matching/Small/Unsmoothed 2                                                                                       
omp sched gather:	kind: OMP_SCHED_DYNAMIC	omp chunkSize: 1	monotonic: N
@COARSENING AC: 2744x2744 ---> 343x343	conf grid: 8x8,	NNZ:12642-18032-12642	 AVG_TIMES_ITERATION:25
@computing Sp3MM as pair of SpMM with func:2 at:0x4070d0	[0m
threadNum: 1	ompGridSize: 8x8	timeAvg:4.188559e-03 timeVar:3.536253e-06	timeInternalAvg:3.345853e-03 (overheads ~ 20.119253% tot) timeInternalVar:2.422955e-06 
threadNum: 2	ompGridSize: 8x8	timeAvg:4.188559e-03 timeVar:3.536253e-06	timeInternalAvg:3.345853e-03 (overheads ~ 20.119253% tot) timeInternalVar:2.422955e-06 
threadNum: 3	ompGridSize: 8x8	timeAvg:4.188559e-03 timeVar:3.536253e-06	timeInternalAvg:3.345853e-03 (overheads ~ 20.119253% tot) timeInternalVar:2.422955e-06 
threadNum: 4	ompGridSize: 8x8	timeAvg:4.188559e-03 timeVar:3.536253e-06	timeInternalAvg:3.345853e-03 (overheads ~ 20.119253% tot) timeInternalVar:2.422955e-06 
==============================================================================================================
export: GROUP_IMPLEMENTATIONS=[false] -> group several compute entries of the same funcID and ComputeConf
                                         adding (in order) implementations compute times as new csvFields like
                                         timeAvg_funcID0_Conf0, timeAvg_funcID0_Conf1, ... timeAvg_funcID1_Conf0, ...
                                        
                                         PARSED LOG CAN HAVE LESS NUM OF COMPUTE LINES, 
                                         BUT THE SAME INITIAL funcID,ComputeConf lines (smaller groups are padded)
        GROUP_IMPLEMENTATIONS_KEEP_CONST_CCONF: assuming ComputeConf being costant across the imput
                                                ComputeConf fields will not be replaced by GROUPD Macro
        FLOAT_PRECISION_PY=[e.g. .17e -> precision of double to output in the csv]
        
usage <logFile>
"""
from collections import namedtuple
from re import finditer
from sys import argv,stderr
from os          import environ as env

GROUP_IMPLEMENTATIONS= "T" in env.get("GROUP_IMPLEMENTATIONS","F").upper()
GROUP_IMPLEMENTATIONS_KEEP_CONST_CCONF = "T" in env.get("GROUP_IMPLEMENTATIONS_KEEP_CONST_CCONF","F").upper()
FLOAT_PRECISION_PY=env.get("FLOAT_PRECISION_PY",".17e")

_FIELDS_MAIN = "source,funcID,timeAvg,timeVar,timeInternalAvg,timeInternalVar,srcSize,dstSize,NNZ_R,NNZ_AC,NNZ_P,sampleSize"
MAIN_FIELDS  = _FIELDS_MAIN.split(",")
#compute config for iterative run as optional fields (requires new python)
_FIELDS_OPT_OMP  = "ompSched,threadNum,ompGrid,buildConf"
_FIELDS_OPT_CUDA = "blockSize_x,blockSize_y,blockSize_z,gridSize_x,gridSize_y,gridSize_z"
_FIELDS_OPT      = _FIELDS_OPT_OMP # + "," + _FIELDS_OPT_CUDA
FIELDS           = _FIELDS_MAIN + "," + _FIELDS_OPT
Execution   = namedtuple("Execution",FIELDS) #,defaults=[None]*len(_FIELDS_OPT.split(",")) ) #require new python
ComputeConf = namedtuple("ComputeConf","funcID,"+_FIELDS_OPT)

GROUPD = "GRPD_ENTRY"   #entry that has been groupped (e.g. because of GROUP_IMPLEMENTATIONS=T)
PADD   = "PADD" #None
GROUP_IMPLEMENTATIONS_TRGT_FIELDS = ["timeAvg"] #,"timeVar"] #fields to "multiplex"
#aux ComputeConf fields selection for GROUP_IMPLEMENTATIONS 
_none           = lambda l: []
_identity       = lambda l: l
_ompGrid        = lambda l: [l[4]]
selectFieldsToAdd = _none

filterFuncLHeader = lambda l: "func:" in l
filterCompLines   = lambda l: all([f in l for f in MAIN_FIELDS[2:6]])

hasFields = lambda l,fNum=2: len(l.strip().split()) > fNum or len(l.strip().split("/")) > fNum
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

GRID_PATTERN="\d+x\d+"
SIZE_PATTERN=GRID_PATTERN+"-\d+NNZ-\d+=MAX_ROW_NZ"
parseGridPattern = lambda s: s.strip().split("x")
FP_PATTERN="[-+]?\d+\.?\d+e[-+]\d+"
parseSizes=lambda s:  s #[int(x) for x in s.split("x")] #TODO not good CSV PARSED

BuildConf = namedtuple("BuildConf","symbUBAssignType bitmapLimbSize")
def parseConfig(confLine):
    srcSize  = parseSizes(getReMatch("COARSENING AC:\s*("+GRID_PATTERN+")",confLine))
    dstSize  = parseSizes(getReMatch("-->\s*("+GRID_PATTERN+")",confLine))
    sampleSize          = int(getReMatch("AVG_TIMES_ITERATION:(\d+)",confLine))
    parallGridSize      = parseSizes(getReMatch("grid:\s*("+GRID_PATTERN+")",confLine))
    nnz_racp            = getReMatch("NNZ:\s*(\d+-\d+-\d+)",confLine)
    nnz_r,nnz_ac,nnz_p  = [int(x) for x in nnz_racp.split("-")]
    #get current build config
    symbUBAssignType    = getReMatch("symbUBAssignType:(\w+)",confLine)
    bitmapLimbSize      = int(getReMatch("bitmapLimbSize:(\d+)",confLine))

    return parallGridSize,srcSize,dstSize,nnz_r,nnz_ac,nnz_p,sampleSize,\
      BuildConf(symbUBAssignType,bitmapLimbSize)

FuncID = namedtuple("FuncID","implNum sp3MMComboType symbType symbAccType")
def parseComputeFuncID(l):
    #gather func identifiers + its subversion
    implNum = int(getReMatch("func:\s*(\d+)",l))
    sp3MMComboType = "pairMuls"
    if "direct" in l:   sp3MMComboType = "direct"
    symbType = "upperBounded"
    if "SymbolicAccurate" in l: symbType = "symbAcc"
    #symbAccType
    try:    symbAccType = getReMatch("with (\w+) ",l)
    except: symbAccType = None

    return FuncID(implNum, sp3MMComboType, symbType, symbAccType)

OmpSched = namedtuple("OmpSched","schedKind chunkSize monotonic fairChunkFolding")
def parseOmpRuntimeSchedule(l):
    schedKind           = getReMatch("kind:\s*(OMP_.+)\s+omp chunk",l)
    chunkSize           = int(getReMatch("omp chunkSize:\s*(\d+)",l))
    monotonic           = "Y" in getReMatch("monotonic:\s*(.)",l)
    fairChunkFolding    = int(getReMatch("fairChunkFolding:(\d+)",l))
    return OmpSched(schedKind,chunkSize,monotonic,fairChunkFolding)

###main parse from here
def parseSingleRun(l): 
    """
    parse single execution (along with its configuration) of a function
    """
    threadNum,ompGridSize=0,[None,None]     #expected 2D  ompGridSize
    if "omp" in l:
            threadNum = getReMatch("threadNum:\s*(\d+)",l)
            ompGridSize = parseSizes(getReMatch("ompGridSize:\s*("+GRID_PATTERN+")",l))
            ompGridSize = parseGridPattern(ompGridSize)
    cudaBlkSize,cudaGridSize = [None]*3,[None]*3
    if "cuda" in l:
            cudaBlkSize   = getReMatch("cudaBlockSize:\s*(\d+\s+\d+\s+\d+)",l).split()
            cudaGridSize  = getReMatch("cudaGridSize:\s*(\d+\s+\d+\s+\d+)",l).split()
    timeAvg = float(getReMatch("timeAvg:\s*("+FP_PATTERN+")",l))
    timeVar = float(getReMatch("timeVar:\s*("+FP_PATTERN+")",l))
    timeInternalAvg = float(getReMatch("timeInternalAvg:\s*("+FP_PATTERN+")",l))
    timeInternalVar = float(getReMatch("timeInternalVar:\s*("+FP_PATTERN+")",l))
    return timeAvg,timeVar,timeInternalAvg,timeInternalVar,\
      threadNum,ompGridSize,cudaBlkSize,cudaGridSize

def parseFuncsExes(lGroup,confLine,ompSched,src):
    """
    parse lines related to diff configurations executions of a function in @lGroup
    composed of a series of (at least 2) lines, an headerLine (funcID) and compute times (times and conf)
    @computing SpMV with func:ID FUNC_ID_STR at:.....   (header)
    [threadNum=..,cudaBlkSize,...],timeAvg,....         (compute lines)
    [threadNum=..,cudaBlkSize,...],timeAvg,....
    
    @Returns:   list of Execution namedtuple of parsed lines @lGroup
    """
    out = list()
    exeConf = parseConfig(confLine)
    parallGridS,matSizes,sampleSize,buildConf=exeConf[0],exeConf[1:-2],exeConf[-2],exeConf[-1]
    ompSched = parseOmpRuntimeSchedule(ompSched)

    for compLinesFuncGroup in filter(filterFuncLHeader,lGroup): #parse single execution log line
        compLinesFuncG = compLinesFuncGroup.split("\n")
        computesFuncID,computesTimes = compLinesFuncG[0],compLinesFuncG[1:]
        funcID = parseComputeFuncID(computesFuncID)
        for l in filter(filterCompLines,computesTimes): #hasFields
            tAvg,tVar,tIntAvg,tIntVar,threadN,ompGridSize,cudaBlkSize,cudaGridSize=parseSingleRun(l)

            isCudaEntry = None not in cudaBlkSize and None in ompGridSize
            if isCudaEntry: #obfuscate dflt redundant prints 
                ompSched        = [None] * len(ompSched)
                threadN         = None
                tIntAvg,tIntVar = None,None #TODO mesured kernel time only

            #insert parsed compute entries infos
            out.append(Execution(src,funcID,tAvg,tVar,tIntAvg,tIntVar,*matSizes,sampleSize,\
              ompSched,threadN,parallGridS,buildConf))
              #*cudaBlkSize,*cudaGridSize)) #TODO CUDA FIELDS
    return out

def groupExesExtendPivot(executionTimes,fieldsToExpand=GROUP_IMPLEMENTATIONS_TRGT_FIELDS): #TODO EXTEND
    """
       given the Execution named tuples in @executionTimes
       group them in a single entry, keeping the info of the funcID and computeConf

       @fieldsToExpand: list of fields to expand with the output of each different tuple in @executionTimes value
       [[[so group tuples by every field not in fieldsToExpand, in particular (optional) computeConf fields]]]

       :Returns only 1, merged, Execution namedtuple where the given @fieldsToExpand will be
                lists of ( ComputeConf,value)
    """
    trgtFieldsGroups = { trgtF:list() for trgtF in fieldsToExpand } #fieldExpandedID:[[ComputeConf,value],...]
    mainFixdFields = executionTimes[0][:len(MAIN_FIELDS)]
    optCConfFieldsFixFirst = executionTimes[0][len(MAIN_FIELDS):]
    for e in executionTimes:
        groupdFields = ComputeConf(e.funcID, *e[len(MAIN_FIELDS):])
        #gather fieldsToExpand of each compute, along with its context info
        for trgtF in fieldsToExpand:    
            trgtFieldsGroups[trgtF].append( [groupdFields, getattr(e,trgtF)] )
    #get the output namedtuple as a "blank" entry, setting the main fields of the first entry
    #andy a the constant GROUPD for every other fields, that has been groupped in the out entry
    optCConfFields = [GROUPD]*len(_FIELDS_OPT.split(","))
    if GROUP_IMPLEMENTATIONS_KEEP_CONST_CCONF: optCConfFields = optCConfFieldsFixFirst
    out = Execution(*mainFixdFields,*optCConfFields)
    out = out._replace(funcID=GROUPD)
    #set the target,groupped fields in the output entry
    for trgtF in fieldsToExpand:    out = out._replace( **{trgtF:trgtFieldsGroups[trgtF]} )
    
    return out

   
def parseLog(log,groupImplementations=GROUP_IMPLEMENTATIONS):
    """
    extract Execution tuples from the string @log
    eventually appling pivots if @groupImplementations....
    """
    executionTimes = list() #Execution tups
    matrixGroup = log.split("#")        #different run of the test program blocks
    linesGroup  = [ g.split("@") for g in matrixGroup[1:] ] #matrixGroup computational lines[threads exes]
    for i,mGroup in enumerate(linesGroup):
        head,computes = mGroup[0],mGroup[1:] #separate header information from a run infos groups
        #parse header informats
        mg = head.split("\n")
        if len(mg) < 2:  print("not complete mGroup",i,mGroup,file=stderr);continue
        header,ompSched,confLine = mg[0],mg[1],mg[2]
        src = header.replace(" ","_").split("/")[-1]
        #parse compute lines
        computeEntries = parseFuncsExes(computes,confLine,ompSched,src) #Execution tuples
        #merge all Execution tuples with a common config in a single one (threading tests)
        if groupImplementations:   computeEntries=[groupExesExtendPivot(computeEntries)]

        executionTimes += computeEntries
    return executionTimes

def csvFormatExpandedTuple(executionTimes):
    #audit data as CSV
    ##RBcomputeEntries have for each fieldsToExpand lists of ( funcID,ComputeConf,time )
    out = ""
    #get largest expansion in all tuples by the first expanded field 
    _trgtF_0 = GROUP_IMPLEMENTATIONS_TRGT_FIELDS[0]
    _largestGrppdEntryLen,largestGrppdEntryIdx \
        = max((len(getattr(e,_trgtF_0)),i) for i,e in enumerate(executionTimes))
    largestGrppdEntry = executionTimes[largestGrppdEntryIdx]
    ##expanded header prepare
    #get expanded fields name suffixes
    csvMultiplexedFiledsSufx = list()
    for funcID,computeConf,_t in getattr(largestGrppdEntry,_trgtF_0):
        cconfCSVfields = selectFieldsToAdd(computeConf)
        csvMultiplexedFiledsSufx.append("_".join([str(x) for x in (funcID,*cconfCSVfields)]))
    #extend header field
    multiplexedCSVHeader = str()
    for f in MAIN_FIELDS:
        if f in GROUP_IMPLEMENTATIONS_TRGT_FIELDS:
            f = ", ".join([f+"_"+suffx for suffx in csvMultiplexedFiledsSufx ])
        multiplexedCSVHeader    += f+", "
    out += str(multiplexedCSVHeader[:-2]) + _FIELDS_OPT #remove last ","
    ##TODO reinforced checks .. used to be 
    ##get a dummy entry to pad Execution entries with target fields with less (multiplexed) values then the max
    ##padEntryV = ((funcID,cconf,None) for funcID,cconf,_t in getattr(largestGrppdEntry,_trgtF_0))
    #padEntryV = getattr(largestGrppdEntry,_trgtF_0)
    #for i in range(len(padEntryV)): padEntryV[i][-1] = PADD

    #dump csv rows, (TODO NOT padding entries with less multiplexed entries)
    for e in executionTimes:    
        for f,x in e._asdict().items():
            if f in GROUP_IMPLEMENTATIONS_TRGT_FIELDS:
                toPadN = _largestGrppdEntryLen - len(x)
                ##if toPadN > 0:  x += padEntryV[len(x):]
                #here select only the times of the groupped(context-ed) values
                assert toPadN == 0
                x = ",".join([str(xx[-1]) for xx in x])
            out += str(x)+", "
        out += "\n"
    return out

if __name__ == "__main__":
    if len(argv) < 2 or "-h" in argv[1]:  print(__doc__);exit(1)
    outCsv = argv[1]+".csv"
    with open(argv[1]) as f:    log=f.read()
    executesTuples = parseLog(log)
    assert len(executesTuples) > 0,"nothing parsed:("
    out = ""
    if not GROUP_IMPLEMENTATIONS:   #just print different tuples formatted
        out += FIELDS+"\n"
        for e in executesTuples:       #dump csv rows
            for f in e: out +=str(f)+", "
                #if type(f) == float:    print(format(f,FLOAT_PRECISION_PY),end=", ")
            out += "\n"
    else: out = csvFormatExpandedTuple(executionTimes)
    
    with open(outCsv,"w") as f:    f.write(out)
    print(out)
