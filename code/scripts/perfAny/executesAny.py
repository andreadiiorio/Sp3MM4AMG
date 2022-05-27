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
pandas based analisis over Sp3MM parsed logs
mainParsedFields
source,funcID,timeAvg,timeVar,timeInternalAvg,timeInternalVar,srcSize,dstSize,NNZ_R,NNZ_AC,NNZ_P,sampleSize,ompSched,threadNum,ompGrid,buildConf
================================================================================
export:     ...outMetrics...
usage:      <flopsLog,log0,log1,....>
"""

from re             import finditer
from sys            import argv,stderr
from os             import environ as env
from collections import namedtuple

from parseLogSp3MM  import *    #parseLog

from pandas import  *
import matplotlib
import matplotlib.pyplot as plt

PLOTSIZE=(15,15)
#PYPLOT AUTOSIZEING
plt.rcParams['axes.xmargin'] = 0; plt.rcParams['axes.ymargin'] = 5e-2   #smart margins for rounding ticks
plt.rcParams['axes.autolimit_mode'] = 'round_numbers' #make last ticks end with figure rounding in case
matplotlib.rcParams.update({'figure.autolayout': True})
baseTrgtColsToPlot = "GFlops serialGFlops"  #GFlopsInternal

###PANDAS UTILS
def filterMaxThreadRows(df,copy=True):
    maxThreadN = max(df["threadNum"])
    if copy:    df = df.copy()
    df[ df["threadNum" == maxThreadN] ]
    return df

def _checkDups(df):
    #dbg unique-ness
    #for i,k in enumerate(df.index.tolist()): print(i,k)
    #for i,k in enumerate(df.index.drop_duplicates().tolist()): print(i,k)
    print("unique idxs len:",len(filterDupsIdx(df)))
    print("unique rows len:",len(filterDupsRow(df)))

##pandas wrappers
filterDupsRow       = lambda df: df.drop_duplicates()
filterDupsBest      = lambda df: df.loc[ df.groupby(df.index, sort=False)['GFlops'].idxmax() ]
filterDupsBest      = lambda df: df.loc[ df.groupby(df.index, sort=False)['timeAvg'].idxmin() ]
filterDupsIdxFirst  = lambda df: df.groupby(df.index).first()
filterDupsIdxLast   = lambda df: df.groupby(df.index).last()
dropLastsRow        = lambda df,n=1,inplace=True:  df.drop(df.tail(n).index,inplace=inplace) # drop last n rows
dropFirstsRow       = lambda df,n=1,inplace=True:  df.drop(df.head(n).index,inplace=inplace)
##mathplotlib wrappers
def setPlotLabels(plot,x,y="E[ GFlop/s ]"):
    plot.set_xlabel(x)
    plot.set_ylabel(y)
def saveAndShowPlot(dfPlot,outFname):
    dfPlot.get_figure().savefig(outFname)
    dfPlot.get_figure().show()


###############################################################################


#def addInputClass(df):    df["inputClass"] = df["source"].apply(classifySrcMatrixes) example of derivate a new column, moved to parser
###### groupBy aux functions
def _testByFunc(arg):
    return arg
def getUBTypeDimensionality(key): #put togheter dimensionality in funcID and UBType from build conf given an index
    funcID    = key[MultiIdxHierarchyFieldsOrder["funcID"]]
    buildConf = key[MultiIdxHierarchyFieldsOrder["buildConf"]]
    #return (buildConf.symbUBAssignType,getDimensionality(funcID)) TODO same size of key... multilevel+func not work :(
    return tuple([(buildConf.symbUBAssignType,getDimensionality(funcID))]+[None]*(len(key)-1))

FuncIDMacroKind = namedtuple("FuncIDMacroKind","dimensionality sp3MMComboType symbType")
FuncIDMacroKind_ = namedtuple("FuncIDMacroKind","sp3MMComboType symbType")

def getMacroKind(funcID):
    dim         = getDimensionality(funcID)
    sp3MMCombo  = funcID.sp3MMComboType
    symbType    = funcID.symbType
    return FuncIDMacroKind(dim,sp3MMCombo,symbType)

def getSrcInputClass(src):
    #src     = key [MultiIdxHierarchyFieldsOrder["source"]]
    return (src.AMG_Method,src.smoothness)
def addDimensionalityCol(df):
    df["DIMs"] = df["funcID"].apply(getDimensionality)
    return df
############################# QUERY  ##########################################
timeFieldsTrgt = "timeAvg timeInternalAvg".split()
BASE_,BASE_DIM,BASE_OMPSCHED_DIM_OMPSCHED = "BASE","BASE_DIM","BASE_OMPSCHED_OMPSCHED"
GROUPING_PARAMS = [BASE_,BASE_DIM,BASE_OMPSCHED_DIM_OMPSCHED]

################## Q1 ##################
def filter2SpMMUB(df):
    #discard symbAccurate and sp3MM direct (since only dyn so no counterpart with static assign yet:( )
    toGroup = df[ (df["funcID"].str[FuncIDOrder["sp3MMComboType"]] == SPMM) & \
                  (df["funcID"].str[FuncIDOrder["symbType"]] == UB) ]
    assert len(toGroup) > 0,"filtered all :("
    return toGroup


def bestUBTmpSpaceAssign(df,grpKind = BASE_DIM):   #Q1
    toGroup = filter2SpMMUB(df)
    assert grpKind in GROUPING_PARAMS, "invalid groupping kind given"
    toGroup["BASE_KIND"] = toGroup["buildConf"].apply(lambda bldConf: bldConf.symbUBAssignType)
    toGroup = addDimensionalityCol(toGroup)
    if grpKind == BASE_ :                     groups = toGroup.groupby("BASE_KIND")
    if grpKind == BASE_DIM :                  groups = toGroup.groupby("BASE_KIND DIMs".split())
    if grpKind == BASE_OMPSCHED_DIM_OMPSCHED: groups = toGroup.groupby("BASE_KIND DIMs ompSched".split())
    #TODO old groupby without any added columns for UB_DIM :)
    #groups = toGroup.groupby(_testByFunc,axis="index",level="_funcID _buildConf".split()) #TODO MULTIPLE LEVEL DON'T WORK!
    #groups = toGroup.groupby(getUBTypeDimensionality,axis="index",as_index=True)
    return groups,toGroup

def _bestUBTmpSpaceAssign2StepsGroupping(df):   #Q1 -- UB_DIM -- old...
    toGroup = filter2SpMMUB(df)
    groups = toGroup.groupby(lambda b: b.symbUBAssignType,axis="index",level="_buildConf",as_index=True)
    assert(len(groups)) == 2,len(groups)
    #extract groups as dfs for specific any on each of them
    dfUBApprochs        = { k:groups.get_group(k) for k in groups.groups.keys() }
    dfUBApprochsMeans   = { k:None for k in groups.groups.keys() }
    for ubType,df in dfUBApprochs.items():
        assert len(df)>0
        trgtFuncsGroups = df.groupby(getDimensionality, axis="index",level="_funcID")
        dfUBApprochsMeans[ubType] = trgtFuncsGroups[timeFieldsTrgt]
    return concat(dfUBApprochsMeans.items())

################## Q2 ##################

def bestIDXMapLimbSize(df,grpingKind = BASE_):     #Q2
    assert grpingKind in GROUPING_PARAMS,"invalid grouping kind give:("
    #get only IDXMAP implementations runs
    toGroup = df[ (df["funcID"].str[FuncIDOrder["symbType"]] == SYMBACC) & \
                  (df["funcID"].str[FuncIDOrder["symbAccType"]] != "RBTREE") ]
    toGroup = addDimensionalityCol(toGroup)

    #differences among IDXMAP runs are only in limbSize
    if grpingKind == BASE_:                       groups = toGroup.groupby("_buildConf")
    if grpingKind == BASE_DIM:                    groups = toGroup.groupby("_buildConf DIMs".split())
    if grpingKind == BASE_OMPSCHED_DIM_OMPSCHED:  groups = toGroup.groupby("_buildConf DIMs ompSched".split())
    return groups,toGroup

################## Q3 ##################


def _filterDirectForbestImplForInputClass(df,ubTmpSpaceAssign,limbSize):    #TODO BROKEN
    toGroup = df[
        (
            (
                (df["funcID"].str[FuncIDOrder["symbType"]] == UB) &
                (
                    (df["buildConf"].str[BuildConfIDOrder["symbUBAssignType"]] == ubTmpSpaceAssign) |
                    (df["funcID"].str[FuncIDOrder["sp3MMComboType"]] == DIRECT)
                )
            ) |
                (
                    (df["funcID"].str[FuncIDOrder["symbType"]] == SYMBACC) &
                    (df["buildConf"].str[BuildConfIDOrder["bitmapLimbSize"]] == limbSize)
                )
        )
    ]
    assert len(toGroup) > 0,"nothing left"
    return toGroup

def filterDirectForbestImplForInputClass(df,ubTmpSpaceAssign,limbSize):    #Q3
    def toKeep(row):
        if   row.funcID.symbType == UB:
            if row.funcID.sp3MMComboType == DIRECT or row.buildConf.symbUBAssignType == ubTmpSpaceAssign:
                return True
        elif row.funcID.symbType == SYMBACC:
            if row.buildConf.bitmapLimbSize == limbSize or row.funcID.symbAccType != "IDXMAP":
                return True
        return False
    toKeepBoolDf = df.apply(toKeep,axis="columns")
    toGroup = df[toKeepBoolDf]
    assert len(toGroup) > 0,"nothing left"
    return toGroup

def bestImplForInputClass(df,ubTmpSpaceAssign,limbSize):    #Q3
    #gather sp3mm or 2spmm with the given @ubTmpSpaceAssign conf and given @limbSize conf
    toGroup = filterDirectForbestImplForInputClass(df,ubTmpSpaceAssign,limbSize)
    #inputClasses  = df.groupby(getSrcInputClass,axis="index",level="_source")
    toGroup = toGroup.copy()
    toGroup["inputClass"] = toGroup["source"].apply(lambda src: (src[0],src[2]) )
    toGroup["implKind"]   = toGroup["funcID"].apply(getMacroKind)
    # toGroup = toGroup[toGroup["ompSched"].str[2] != 2]
    # assert len(toGroup) > 0,"all removed :("
    # #simplify ompSched Col
    # toGroup["ompSched"] = toGroup["ompSched"].map(lambda ompS: ompS[0])
    groups = toGroup.groupby("inputClass implKind ompSched".split())

    return groups,toGroup
################## Q4 ##################
def bestForEachMatrix(df):
    groupsMatrices = df.groupby("_source")
    bestTimesPerMatrix  = groupsMatrices.agg(
        minTime         = ("timeAvg","min"),
        minTimeInternal = ("timeInternalAvg","min"),
        src             = ("source","first"),
        serialTime      = ("serialTimeAvg","first"),
        #minSerialTimeInternal = ("serialTimeAvg","min")
    )
    bestTimesPerMatrixMinTimeIdxs           = groupsMatrices["timeAvg"].idxmin()
    bestTimesPerMatrixMinTimeInternalIdxs   = groupsMatrices["timeInternalAvg"].idxmin()
    bestTimesPerMatrixMinTimeImpl           = df.loc[bestTimesPerMatrixMinTimeIdxs]
    bestTimesPerMatrixMinTimeInternalImpl   = df.loc[bestTimesPerMatrixMinTimeInternalIdxs]
    return  bestTimesPerMatrixMinTimeImpl,bestTimesPerMatrixMinTimeInternalImpl,bestTimesPerMatrix
def f(a):
    return True
def best2DGrid(df):
    toGroup = df[ df["funcID"].str[FuncIDOrder["implNum"]] >= 2 ]   #GET ONLY 2D IMPLEMENTATIONS
    assert len(toGroup) > 0,"filtered all :("
    groups = toGroup.groupby("funcID ompGrid".split())
    #TODO GATHER LARGEST SET OF MATRIXES COMMON TO ALL GROUPS FOR FILTERING AGAIN toGruop then do the means
    groups.aggregate(f)
    return groups,toGroup

def bestImplVersPerMatrix(df,topN=3):
    """
    among all implementations in Dataframe @df, find the best @topN ones for each matrix group
    :return: list of tuples: (matrixGroup,[bestImplementation_row_0th,1th,...topN-th])
    """
    df.groupBy()


###############################################################################


MultiIdxHierarchyFields = "source funcID ompGrid ompSched buildConf threadNum label".split()
MultiIdxHierarchyFieldsOutNames = ["_"+f for f in MultiIdxHierarchyFields]
MultiIdxHierarchyFieldsOrder = {f:i for i,f in enumerate(MultiIdxHierarchyFields)}

def addFlopsSerialCols(executionDf,flopsDf):
    flopsToAddColsDf = flopsDf["flopN serialTimeAvg serialTimeVar serialGFlops".split()]
    executionDfWithFlops = executionDf.merge(flopsToAddColsDf,left_on="_source",right_index=True,sort=False) #keys inner join
    executionDfWithFlops["GFlops"] = executionDfWithFlops["flopN"] / (executionDfWithFlops["timeAvg"] * 1e9)
    executionDfWithFlops["GFlopsInternal"] = executionDfWithFlops["flopN"] / (executionDfWithFlops["timeInternalAvg"] * 1e9)

    return executionDfWithFlops

def fullConfigurationAny(args):
    l = len(args)
    assert l >= 2 , __doc__
    with open(args[0]) as f:    logFlopsTuples = parseFlopsLog(f.read().strip())
    logFlopsDf = DataFrame(logFlopsTuples)
    logFlopsDf["source"] = logFlopsDf["source"].map(srcReplaceMtxWithLev)
    logFlopsDf.index = Index(logFlopsDf["source"]).rename("_source")
    logFlopsDf["serialGFlops"] = logFlopsDf["serialFlops"] / 1e3
    serialGflopsAvgAll = logFlopsDf["serialGFlops"].mean()
    logsDFs = list()
    for x in range(1,l):
        logFname = args[x]
        logLabel = logFname[:logFname.rfind(".")]
        with open(logFname) as f:   logExecutesTuples = parseExecutionsLog(f.read())
        if len(logExecutesTuples) == 0: print(logFname,"no tuples founded...",file=stderr);continue
        logEntriesDF                = DataFrame(logExecutesTuples)
        logEntriesDF["source"] = logEntriesDF["source"].map(srcReplaceMtxWithLev)
        #SIMPLIFY DF FOR QUERIES
        logEntriesDF["ompSched"] = logEntriesDF["ompSched"].map(lambda omps: omps.schedKind )
        logEntriesDF["ompGrid"] = None
        #add the given label to the parsed df
        logEntriesDF.label = logLabel
        logEntriesDF["label"] = logLabel
        #move from default index to defining columns
        logEntriesDF.index = MultiIndex.from_frame(logEntriesDF[MultiIdxHierarchyFields]).rename(MultiIdxHierarchyFieldsOutNames)
        logsDFs.append(logEntriesDF)
        print("parsed log into dataframe of",len(logEntriesDF),"idx unique:",logEntriesDF.index.is_unique)
    #toAny = concatDFs(logsDFs)
    _lastDF = logEntriesDF
    dfAll = concat(logsDFs)
    if not dfAll.index.is_unique:
        l = len(dfAll)
        dfAll = filterDupsBest(dfAll)
        print("dups indexes present... dropped",l-len(dfAll))

    dfAll = addFlopsSerialCols(dfAll,logFlopsDf)
    dfAllLen,dfAllCp = len(dfAll),dfAll.copy()

    ubTmpSpaceImplGroup,ubTmpSpaceImplSrcs = bestUBTmpSpaceAssign(dfAll)                    #Q1
    idxmapLimbImplGroup,idxmapLimbImplSrcs = bestIDXMapLimbSize(dfAll)                      #Q2
    #best2DGridGroup,best2DGridSrcs = best2DGrid(dfAll)  #Q5_0
    inClassesImplGroup,inClassesImplSrcs   = bestImplForInputClass(dfAll,"DYN_ASSIGN",128)  #Q3
    inClassesImplMeans = inClassesImplGroup.mean()
    print(inClassesImplMeans)
    #q3 = inClassesImplMeans["GFlops"].unstack(0);

    # inClassesImplMeans["implKind"] = inClassesImplMeans.index.get_level_values("implKind")
    # inClassesImplMeans["inputClass"] = inClassesImplMeans.index.get_level_values("inputClass")
    # inClassesImplMeans["ompSched"] = inClassesImplMeans.index.get_level_values("_ompSched")
    # inClassesImplMeans.pivot(index="inputClass", columns="implKind", values="GFlops")
    def q3MapIdxLabel(k):
        inClass,fMacroID,ompSched = k
        if fMacroID.sp3MMComboType == DIRECT:   f = "Sp3MM_DIRECT_UB"
        else:   f = "2*SpMM_"+fMacroID.symbType
        return f,inClass,ompSched #.schedKind.replace("_SCHED","")
    q3_xtickLabels = [ q3MapIdxLabel(k) for k in inClassesImplGroup.groups.keys()]

    bestTimesPerMatrixMinTimeImpl,bestTimesPerMatrixMinTimeInternalImpl,bestTimesPerMatrix =  bestForEachMatrix(dfAll) #Q4
    q4Base = bestTimesPerMatrixMinTimeImpl.droplevel(bestTimesPerMatrixMinTimeImpl.index.names[1:])

    #q4Base.index = q4Base.index.map(srcReplaceMtxWithLev)
    ###PLOTS
    #Q1

    q1 = ubTmpSpaceImplGroup.mean()["GFlops"];
    q1.loc["serial"] = serialGflopsAvgAll
    q1Plot = q1.plot.bar();
    setPlotLabels(q1Plot,"Configurazioni SpMM UpperBound")
    print("\n".join(dir(q1Plot)),"\n".join(dir(q1Plot.get_figure())),sep="\n\n")
    #q1Plot.set_xticklabels(list(range(len(q1Plot.get_xticklabels()))))
    saveAndShowPlot(q1Plot,"q1.svg")
    assert len(dfAll) == dfAllLen
    #Q2

    q2 = idxmapLimbImplGroup.mean()["GFlops"];
    q2.loc["serial"] = serialGflopsAvgAll
    q2Plot = q2.plot.bar()
    setPlotLabels(q2Plot,"Configurazioni dimensioni limb")
    q2Plot.set_xticklabels(["64bit","128bit","serial"]) #list(range(len(q1Plot.get_xticklabels()))))
    saveAndShowPlot(q2Plot,"q2.svg")

    assert len(dfAll) == dfAllLen

    #Q3

    q3 = inClassesImplMeans["GFlops"].unstack("implKind ompSched".split());
    q3SmothedClass   = q3.loc[ q3.index.get_level_values("inputClass").str[1] == "Smoothed"]
    q3UnsmothedClass   = q3.loc[ q3.index.get_level_values("inputClass").str[1] == "Unsmoothed"]

    q3SmothedClassPlot = q3SmothedClass.plot.bar()
    setPlotLabels(q3SmothedClassPlot,"Performance implementazioni Sp[3]MM per classe di input")
    #q3Plot.set_xticklabels(q3_xtickLabels)
    q3SmothedClassPlot.autoscale(True,axis="y")
    print(q3SmothedClassPlot.get_figure().get_figheight(),q3SmothedClassPlot.figbox)
    q3SmothedClassPlotFig = q3SmothedClassPlot.get_figure()
    q3SmothedClassPlotFig.set_figheight(q3SmothedClassPlotFig.get_figheight()+4)
    saveAndShowPlot(q3SmothedClassPlotFig,"q3-smooth.svg")

    q3UnmothedClassPlot = q3UnsmothedClass.plot.bar()
    setPlotLabels(q3UnmothedClassPlot,"Performance implementazioni Sp[3]MM per classe di input")
    #q3Plot.set_xticklabels(q3_xtickLabels)
    q3UnmothedClassPlot.autoscale(True,axis="y")
    print(q3UnmothedClassPlot.get_figure().get_figheight(),q3UnmothedClassPlot.figbox)
    q3UnmothedClassPlotFig = q3UnmothedClassPlot.get_figure()
    q3UnmothedClassPlotFig.set_figheight(q3UnmothedClassPlotFig.get_figheight()+4)
    saveAndShowPlot(q3UnmothedClassPlotFig,"q3-unsmoot.svg")

    assert len(dfAll) == dfAllLen
    #Q4
    q4Base.sort_values(by="GFlops",ascending=False,inplace=True)
    #better view
    dropLastsRow(q4Base,4,True)
    q4Small = dropFirstsRow(q4Base,10,False)
    q4Big = q4Base.head(10)

    q4BigPlot = q4Big[baseTrgtColsToPlot.split()].plot.bar()
    setPlotLabels(q4BigPlot,"Migliore implementazione parallela VS seriale")
    saveAndShowPlot(q4BigPlot,"q4-b.svg")

    q4SmallPlot = q4Small[baseTrgtColsToPlot.split()].plot.bar()
    setPlotLabels(q4SmallPlot,"Migliore implementazione parallela VS seriale")
    saveAndShowPlot(q4SmallPlot,"q4-s.svg")

    plt.show()
    exit(0)
    #audit some summary parsing infos
    # _lastDfWithFlops = addFlopsSerialCols(_lastDF,logFlopsDf)
    # srcsMatrixesGroups  = _lastDF["source"].sort_values().drop_duplicates(inplace=False)
    # print("srcsMatrixesGroups",srcsMatrixesGroups.array)
    # print("last inserted df len",len(logEntriesDF),"dfAllLen",len(dfAll),"dfAll idx unique:",dfAll.index.is_unique)
    # #add flops to df
    # # _mokkedFlopsDF      = DataFrame(srcsMatrixesGroups)
    # # _mokkedFlopsDF["flops"] = 0
    # # _mokkedFlopsDF["serialTime"] = 0
    # # dfAllFlops = dfAll.merge(_mokkedFlopsDF,on="source")

def separateMatrixFuncThreadsGroups(df):
    runGroups = df.groupby("_source _funcID".split())
    runDfs = [runGroups.get_group(gK) for gK in runGroups.groups.keys() ]
    return runDfs

def varThreadNAny(args):    #Q4
    logsDFs = list()
    for logPath in args[1:]:
        with open(logPath) as f:   logExecutesTuples = parseExecutionsLog(f.read())
        logEntriesDF = DataFrame(logExecutesTuples)
        logEntriesDF.label = logPath
        logEntriesDF["label"] = logPath
        logEntriesDF["source"] = logEntriesDF["source"].map(srcReplaceMtxWithLev)

        # move from default index to defining columns
        logEntriesDF.index = MultiIndex.from_frame(logEntriesDF[MultiIdxHierarchyFields])\
            .rename(MultiIdxHierarchyFieldsOutNames)
        logsDFs.append(logEntriesDF)
        print("parsed log into dataframe of", len(logEntriesDF), "idx unique:", logEntriesDF.index.is_unique)
    # toAny = concatDFs(logsDFs)
    lastDF = logEntriesDF
    dfAll = concat(logsDFs)
    with open(args[0]) as f:    logFlopsTuples = parseFlopsLog(f.read().strip())
    logFlopsDf = DataFrame(logFlopsTuples)
    logFlopsDf["source"] = logFlopsDf["source"].map(srcReplaceMtxWithLev)
    logFlopsDf.index = Index(logFlopsDf["source"]).rename("_source")
    logFlopsDf["serialGFlops"] = logFlopsDf["serialFlops"] / 1e3

    dfAll = addFlopsSerialCols(dfAll,logFlopsDf)

    runDfs = separateMatrixFuncThreadsGroups(dfAll)
    for df in runDfs:
        dfSource = df["source"].head(1)[0]
        dfFuncID = mapFuncIDToStr(df["funcID"].head(1)[0])
        matrix = " ".join(dfSource[:-1])+" %d" % dfSource[-1]
        serialGFlops  = df["serialGFlops"].head(1)[0]
        trgt = df["GFlops"]
        trgt.loc["serialGFlops"] = serialGFlops
        p = trgt.plot.bar()
        p.set_xticklabels([str(i) for i in range(40,0,-1)]+["serialGFlops"])
        setPlotLabels(p,"Performance con thread variabili per\n %s e l'implementazione %s" % (matrix,dfFuncID))
        saveAndShowPlot(p,"%s_varThreads.svg" % (matrix+dfFuncID))
    print(runDfs)
    ###Q4
    #dfAll["inputClass"] = dfAll["source"].apply(lambda src: tuple(src[:1]))         #amgMethod and size
    #groups = dfAll.groupby("inputClass funcID ompSched".split())
    #TODO groups stats

if __name__ == "__main__":
    argv.pop(0)
    if len(argv) < 2 or "-h" in argv:  print(__doc__, file=stderr);exit(1)

    if "ANY_MAXTHREAD" in env:      fullConfigurationAny(argv)
    if "ANY_MULTITHREAD" in env:    varThreadNAny(argv)