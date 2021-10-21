"""
Andrea Di Iorio
scan from a base dir for triples or quadruples of file that match this rule:
..../dir/.*l(LEVNUM+1)_r.mtx
..../dir/.*l(LEVNUM)_ac.mtx
..../dir/.*l(LEVNUM+1)_p.mtx
..../dir/.*l(LEVNUM+1)_ac.mtx
these file will rappresent dump of matrixes of AMG coarsening of matrixes 
 from level LEVNUM to LEVNUM+1
lLEVNUM_ac = lLEVNUM_r*l(LEVNUM-1)_ac*lLEVNUM_p
dump each group of matrixes in a  line of list file, that will be easy to trasform in bash script to execute the SP3GEMM over all matrix groups
matrix names parsed with regex groups at: MAT_NAME_PARSE_PATTERN
usage: [baseDirForSearch (dflt=%s),includeNextLevelACInGroups (dflt=%s)]
export: MATRIX_EXT (dflt=.mtx,to identify matrix files);DEBUG (dflt=False);ABSPATH (dflt=True)
"""

from os import walk,path,environ
from sys import argv,stderr
from collections import namedtuple
from re import finditer

MATRIX=namedtuple("MATRIX","levN roleName fullname")
MATRIX_LEVEL_GROUP=namedtuple("MATRIX_LEVEL_GROUP",\
 "r ac p ac_next levN ancestorDir",defaults=[None,None,None])

DEBUG=bool(environ.get("DEBUG",False))
ABSPATH=bool(environ.get("ABSPATH",True ))
ROOTDIR="."
INCLUDENEXTAC=True
MATRIX_EXT=environ.get("MATRIX_EXT",".mtx")
MAT_NAME_PARSE_PATTERN=".*l([0-9]+)_(.+).mtx"
getReGroups=lambda pattern,string:\
    finditer(pattern,string).__next__().groups()

def getPath(currRelDir,fname):
    base = currRelDir
    if ABSPATH: base=path.abspath(currRelDir)
    return path.join(base,fname)


def parseMatName(f):
    fields=getReGroups(MAT_NAME_PARSE_PATTERN,f)
    return MATRIX(int(fields[0]),fields[1],f)

def scanMatrixGroups(baseDir=ROOTDIR):
    groups=list()
    for currRelDir, _dirs, files in walk(baseDir):
        matrixes=list()
        for f in files:
            if MATRIX_EXT in f:   matrixes.append(parseMatName(f))
        if len(matrixes) < 3:   continue #skipping currRelDir
        minLev=min([m.levN for m in matrixes])
        maxLev=max([m.levN for m in matrixes])
        matLevels={m.roleName+str(m.levN):m.fullname for m in matrixes}
        for l in range(minLev,maxLev+1):
            r       = matLevels.get("r"+str(l+1),None)
            ac      = matLevels.get("ac"+str(l),None)
            p       = matLevels.get("p"+str(l+1),None)
            ac_next = matLevels.get("ac"+str(l+1),None)
            if None in [r,ac,p]:    print("#not full group at",currRelDir,l);continue
            if r!=None:         r  = getPath(currRelDir,r)
            if ac!=None:        ac = getPath(currRelDir,ac)
            if p!=None:         p  = getPath(currRelDir,p)
            if ac_next!=None:   ac_next = getPath(currRelDir,ac_next)
            groups.append(MATRIX_LEVEL_GROUP(r,ac,p,ac_next,l,currRelDir))
            #efficient-harder way 
            #for m in filter(lambda m:m.levN==l,matrixes): #TODO FILL A GROUP
    return groups
        
if __name__ == "__main__":
    if "-h" in argv[1]: print(__doc__ % (ROOTDIR,str(INCLUDENEXTAC)));exit(1)
    if len(argv)>1:     ROOTDIR=argv[1]
    if len(argv)>2:     INCLUDENEXTAC=bool(argv[2])
    
    matGroups = scanMatrixGroups(ROOTDIR)
    print("#founded",len(matGroups),"groups",len([None!=m.ac_next for m in matGroups]),"full groups")
    for g in matGroups:
        print("\n",g.r,g.ac,g.p,end="\t")
        if INCLUDENEXTAC:   print(g.ac_next,end="\t")
        print("#",g.ancestorDir,g.levN)
