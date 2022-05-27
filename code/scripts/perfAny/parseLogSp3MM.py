#              Sp3MM_for_AlgebraicMultiGrid
#    (C) Copyright 2021-2022
#        Andrea Di Iorio      
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions, and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. The name of the Sp3MM_for_AlgebraicMultiGrid or the names of its contributors may
#       not be used to endorse or promote products derived from this
#       software without specific written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE Sp3MM_for_AlgebraicMultiGrid GROUP OR ITS CONTRIBUTORS
#  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

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

GRID_PATTERN    = "\d+x\d+"
SIZE_PATTERN    = GRID_PATTERN+"-\d+NNZ-\d+=MAX_ROW_NZ"
FP_EXP_PATTERN  = "[-+]?\d+\.?\d+e[-+]\d+"
FP_PATTERN      = "[-+]?\d+\.\d+"
parseSizes      = lambda s:  s #[int(x) for x in s.split("x")] #TODO not good CSV PARSED
parseGridPattern = lambda s: s.strip().split("x")

BuildConf = namedtuple("BuildConf","symbUBAssignType bitmapLimbSize")
BuildConfIDOrder = {k:i for i,k in enumerate(BuildConf._fields)}

def parseMatSizes(confLine):
    srcSize     = parseSizes(getReMatch("COARSENING AC:\s*(" + GRID_PATTERN + ")", confLine))
    dstSize     = parseSizes(getReMatch("-->\s*(" + GRID_PATTERN + ")", confLine))
    sampleSize  = int(getReMatch("AVG_TIMES_ITERATION:(\d+)", confLine))
    nnz_racp    = getReMatch("NNZ:\s*(\d+-\d+-\d+)", confLine)
    nnz_r_ac_p  = [int(x) for x in nnz_racp.split("-")]
    return srcSize,dstSize,nnz_r_ac_p,sampleSize

def parseConfig(confLine):
    srcSize, dstSize, nnz_r_ac_p, sampleSize = parseMatSizes(confLine)
    parallGridSize = parseSizes(getReMatch("grid:\s*(" + GRID_PATTERN + ")", confLine))

    #get current build config
    symbUBAssignType    = getReMatch("symbUBAssignType:(\w+)",confLine)
    bitmapLimbSize      = int(getReMatch("bitmapLimbSize:(\d+)",confLine))

    return parallGridSize,srcSize,dstSize,*nnz_r_ac_p,sampleSize,\
      BuildConf(symbUBAssignType,bitmapLimbSize)

Source = namedtuple("Source","AMG_Method size smoothness mtxs",defaults=[None])
def classifySrcMatrixes(srcStr):
    #classify triple product matrix from VanekBrezina/Medium/Smoothed/dump_lev_d_p0_l002_r.mtx to Inputclass
    matricesFields = [m.split("/") for m in srcStr.split()]
    #check other matrixes in srcStr are the same type
    matricesInputClasses = [ Source(*m[:-1]) for m in matricesFields ] #just take the input class of each matrix
    m_r = matricesInputClasses[0]
    for m in matricesInputClasses[1:]:  assert( m == m_r)       #TODO validation srcStr matrixes of same class!
    return m_r._replace(mtxs=tuple([mFields[-1] for mFields in matricesFields])) #add mtxs directly
def srcReplaceMtxWithLev(source):
    levStart = source.mtxs[1].find("00")
    lev = int(source.mtxs[1][levStart:levStart + 3])
    return source._replace(mtxs=lev)

##FUNC ID CONSTS -runtime part-
FuncID = namedtuple("FuncID","implNum sp3MMComboType symbType symbAccType")
FuncIDOrder = {f:i for i,f in enumerate(FuncID._fields)}
FUNC_ID_DIM = {0:"1D",1:"1D",2:"2D",3:"2D"}
def getDimensionality(funcID):    return FUNC_ID_DIM[funcID.implNum]

def mapFuncIDToStr(funcID):
    dim = getDimensionality(funcID)
    pref = funcID.symbType
    if funcID.symbAccType != "None":          pref+="_"+funcID.symbAccType
    elif funcID.sp3MMComboType == DIRECT:   pref+="_direct"
    return pref+"_"+str(dim)

UB,SYMBACC  =   "UpperBound","SymbolicAccurate"     #symb phase kind
SPMM,DIRECT =   "pairMuls","direct"                 #composition of Sp3MM kind
def parseComputeFuncID(l):
    #gather func identifiers + its subversion
    implNum = int(getReMatch("func:\s*(\d+)",l))
    sp3MMComboType =    SPMM
    if "direct" in l:   sp3MMComboType = DIRECT
    symbType = UB
    if "SymbolicAccurate" in l: symbType = SYMBACC
    #symbAccType
    try:    symbAccType = getReMatch("with (\w+) ",l)
    except: symbAccType = "None"

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
            threadNum   = int(getReMatch("threadNum:\s*(\d+)",l))
            ompGridSize = parseSizes(getReMatch("ompGridSize:\s*("+GRID_PATTERN+")",l))
            ompGridSize = parseGridPattern(ompGridSize)
    cudaBlkSize,cudaGridSize = [None]*3,[None]*3
    if "cuda" in l:
            cudaBlkSize   = getReMatch("cudaBlockSize:\s*(\d+\s+\d+\s+\d+)",l).split()
            cudaGridSize  = getReMatch("cudaGridSize:\s*(\d+\s+\d+\s+\d+)",l).split()
    timeAvg = float(getReMatch("timeAvg:\s*(" + FP_EXP_PATTERN + ")", l))
    timeVar = float(getReMatch("timeVar:\s*(" + FP_EXP_PATTERN + ")", l))
    timeInternalAvg = float(getReMatch("timeInternalAvg:\s*(" + FP_EXP_PATTERN + ")", l))
    timeInternalVar = float(getReMatch("timeInternalVar:\s*(" + FP_EXP_PATTERN + ")", l))
    return timeAvg,timeVar,timeInternalAvg,timeInternalVar,\
      threadNum,ompGridSize,cudaBlkSize,cudaGridSize

def parseFuncsExes(lGroup,confLine,ompSched,srcs):
    """
    parse lines related to diff configurations executions of a function in @lGroup
    composed of a series of (at least 2) lines, an headerLine (funcID) and compute times (times and conf)
    @computing SpMV with func:ID FUNC_ID_STR at:.....   (header)
    [threadNum=..,cudaBlkSize,...],timeAvg,....         (compute lines)
    [threadNum=..,cudaBlkSize,...],timeAvg,....
    
    @Returns:   list of Execution namedtuple of parsed lines @lGroup
    """
    out = list()
    src = classifySrcMatrixes(srcs)
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

def parseFlopsLine(l):
    flopN          = int(getReMatch("flop:(\d+)", l))
    serialTimeAvg   = float(getReMatch("elapsedAvg:(%s)\t" % FP_PATTERN, l))
    serialTimeVar   = float(getReMatch("elapsedVar:(%s)" % FP_PATTERN, l))
    serialMFlops    = float(getReMatch("MegaflopsAvg:(%s)" % FP_PATTERN, l))
    return  flopN, serialTimeAvg, serialTimeVar, serialMFlops

FlopsSp3MM = namedtuple("Flops","source flopN srcSize dstSize NNZ_R NNZ_AC NNZ_P  AVG_TIMES_ITERATION serialTimeAvg serialTimeVar serialFlops")
def parseFlopsLog(log):
    """
    parse flops program log, like:
    #r ac p ac_next
    COARSENING AC: 2744x2744 ---> 343x343	conf grid: 20x2,	NNZ:12642-18032-12642	 AVG_TIMES_ITERATION:40	symbUBAssignType:STATIC_ASSIGN	bitmapLimbSize:128
    Sp3MM as 2 SpMM	flop:9350720	elapsedAvg:0.004522	elapsedVar:0.000000	MegaflopsAvg:2067.960706
    """
    out = list()
    matrixGroup = [mg.strip().split("\n") for mg in log.split("#")[1:]]       #different runs' lines  of the test program blocks
    for mg in matrixGroup:
        sources,confLine,flopLine = mg
        srcs = " ".join(("/".join(m.split("/")[-4:])  for m in sources.split()))
        srcSize, dstSize, nnz_r_ac_p, sampleSize  = parseMatSizes(confLine)
        flopN,serialTimeAvg,serialTimeVar,serialMFlops      = parseFlopsLine(flopLine)
        out.append(FlopsSp3MM(classifySrcMatrixes(srcs),flopN,srcSize,dstSize,*nnz_r_ac_p,sampleSize,serialTimeAvg,serialTimeVar,serialMFlops))
    return out

def parseExecutionsLog(log, groupImplementations=GROUP_IMPLEMENTATIONS):
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
        srcs = " ".join(("/".join(m.split("/")[-4:])  for m in header.split()))
        #parse compute lines
        computeEntries = parseFuncsExes(computes,confLine,ompSched,srcs) #Execution tuples
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
    executesTuples = parseExecutionsLog(log)
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
