/*
 * dev Andrea Di Iorio
 * draw a [sparse] matrix as an image with a black square dot for each nonzero elem
 * scaling the size of the matrix, "oring" the nz elem in a square of the original matrix into 
 * the smaller destination pixel grid.
 * Highlight nz elem into square of pixel of custom size, that will incorporate near elements
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

#include "macros.h"
#include "sparseMatrix.h"
#include "sparseMatrixToImage.h"
#include "utils.h"
#include "parser.h"

double* parseDenseMatrixPatternFromFile(char* fpath,uint* M,uint* N){
    return NULL;
}
//set @i,@j in the scaled pixel grid into raw PPM -> RGB data as a black dot
static inline void _setBlackDotRGB(uint w,uint h, uchar rawData[][w],int i,int j){
    if  ( i<0 || i> (int)h )        return;
    for ( int jj=j; jj<j+PPM_CHAN_NUM; jj++ ){
        //skip outOfBorder enlarged dots
        if ( jj<0 || jj > (int) w)    continue; 
        //set RGB black -> no intensity in every channel
        rawData[i][jj] = 0;
    }
}

/*
 * set a black dot pixel in MATRIX coordinate @i,@j , that will be enlarged
 * to be more visible to a square of @unifyNearW^2 near pixels 
 * (discarding previous 0 -> white dots, 
 *  NB never setted a white dot, memset at the start)
 *  TODO TODO FIX unifyNearW!!
 */ 
static void setBlackNZPixel(ppmData* data,int i,int j,ushort unifyNearW){
    //set the first dot
    _setBlackDotRGB(data->width*PPM_CHAN_NUM,data->height, 
      (uchar (*)[data->width*PPM_CHAN_NUM]) data->data,i,PPM_CHAN_NUM*(j));
    //set the enlarged dots
    for (short ww,w=0;w<unifyNearW;w++){
        for (short zz,z=0;z<unifyNearW;z++){
            //make the highlight unify square centered in (@i,@j)
            ww = INT_DIV_CEIL(w,2); 
            zz = INT_DIV_CEIL(z,2); 
            if (!(w % 2))   ww *= -1;
            if (!(z % 2))   zz *= -1;

            
            _setBlackDotRGB(data->width*PPM_CHAN_NUM,data->height, 
              (uchar (*)[data->width*PPM_CHAN_NUM]) data->data, 
              i+ww,PPM_CHAN_NUM*(j+zz));
        }
    }
}
void denseMatrixToPPM(ppmData* data,
  uint M, uint N, double mat[][N],ushort step, ushort unifyNearW){
    char nz;    //flag to mark a founded nz
    for (uint i=0;i<M;i+=step){
        for (uint j=0;j<N;j+=step){
            //i,j point to the first: top,left element in the search square
            nz = 0;
            for (uint w=0; w<step && !nz; w++){
                for (uint z=0; z<step && !nz; z++){
                    if (mat[i+w][j+z]){
                        nz = (!0);
                        setBlackNZPixel(data,i/step,j/step,unifyNearW);
                        break;
                    }
                }
            }
        }
    }
}
void sparseMatrixToPPM(ppmData* data,spmat* sparseMat,
    ushort step, ushort unifyNearW){
    ERRPRINT("TODO");
}

//check if MXN dense matrix @mat has a black pixel corresponding for each nonzero element
//TODO add support for step, unifyNearW, NOW CONSIDERED AS dflt... ->1a1 mat
static int checkDenseMatrixToPPM(uint M, uint N,
  unsigned char rawData[][3*N],double mat[][N],ushort step, ushort unifyNearW){
    for (uint i=0; i<N; i++){ 
        for (uint j=0; j<M; j++){
            if (mat[i][j] && 
              (rawData[i][3*j+0] || rawData[i][3*j+1] || rawData[i][3*j+2])){
                fprintf(stderr,"not matching NNZ at (%u,%u)\n",i,j);
                return EXIT_FAILURE;
            }
            else if ( ! mat[i][j] && 
              (rawData[i][3*j+0]!=255 || rawData[i][3*j+1]!=255 || rawData[i][3*j+2]!=255)){
                fprintf(stderr,"not matching ZERO at (%u,%u)\n",i,j);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
#ifdef MAIN_SPMAT_IMG
/*
 * build this file as a stand alone executable, that get the input matrix from a serialized file
 * along with the parameters from argv defining MAT2PPM_STANDALONE
 * otherwise just export the next function
 */
int main(int argc,char** argv){
    void* map = NULL;
    int out=EXIT_FAILURE,outFd=0,mode=S_IRWXU;
    if (argc < 3 )    {
        ERRPRINT("usage: inputMatrixFile, "PATTERN_DENSE" || "MM_COO","
         " [elemSquareWPixel=1, unifyNearW=1, outPPMFPath="DFLT_OUTPATH"]\n");
        return EXIT_FAILURE;
    }
    double* denseMat;
    spmat*  sparseMat;
    ppmData* data=NULL;
    uint M,N;
    if (!strncmp(argv[2],PATTERN_DENSE,strlen(PATTERN_DENSE))){
        if(!(denseMat = parseDenseMatrixPatternFromFile(argv[1],&M,&N))){
            ERRPRINT("dense matrix pattern parse failed\n");
            return EXIT_FAILURE;
        }
    } else if (!strncmp(argv[2],MM_COO,strlen(MM_COO))){
        if (!(sparseMat = MMtoCSR(argv[1]))){
            ERRPRINT("sparse matrix MM coordinate parsing failed\n");
            return EXIT_FAILURE;
        }
        M = sparseMat -> M;
        N = sparseMat -> N;
        if (!(denseMat = CSRToDense(sparseMat))){ //TODO REPLACE WITH _sparseMatrixToPPM LATER
            ERRPRINT("CSRToDense FAILED\n");
            goto _free;
        }
    }else {ERRPRINT("INVALID IN MATRIX FORMAT!\n");return EXIT_FAILURE;}
        
    const char* outFname=DFLT_OUTPATH;
    uint elemSquareWPixel = 1, unifyNearW = 0;
    //if (argc >= 4)    outFname = argv[3]; // PUT OTHER OPTIONS

    data=malloc(sizeof(*data));
    if (!data){
        ERRPRINT("ppmData alloc for dense matrix failed\n");
        return EXIT_FAILURE;
    }
    //uint pad=2*pixelsPaddingPerElem;data- width=ceil(N/elemSquareWPixel)*(1+pad)
    //set out image size considering both scaling and padding
    data -> width  = ceil(N / elemSquareWPixel);
    data -> height = ceil(M / elemSquareWPixel);
    int headerLen = snprintf(data->header,PPM_HEADER_MAX_LEN,
        "P6\n%u %u\n255\n",data->width,data->height); 
    if (headerLen < 0){
        ERRPRINT("snprintf error");
        goto _free; 
    }
    uint pixelDataLen = data->width*data->height*PPM_CHAN_NUM;
    uint dataLen = pixelDataLen + headerLen;
    
    ///out file mmap for easy write
    outFd=open(outFname, O_RDWR | O_CREAT | O_EXCL | O_TRUNC, mode);
    if (errno==EEXIST)     outFd=open(outFname, O_RDWR | O_TRUNC, mode);
    if (outFd<0){
        perror("open outFd failed");
        goto _free;
    }
    if (ftruncate(outFd,dataLen)<0){
        perror("ftruncate err");
        goto _free;
    }
    map = mmap(NULL, dataLen, PROT_WRITE, MAP_SHARED, outFd, 0);    
    if (map == MAP_FAILED){
        perror("mmap failed... ");
        goto _free;
    }

    memcpy(map,data->header,headerLen);        //write header
    memset(map+headerLen,255,pixelDataLen);    //preset the img as white
    data->data=map+headerLen; //directly write converted matrix to outfile via mmap
    denseMatrixToPPM(data,M,N,(double (*)[N]) denseMat,elemSquareWPixel,unifyNearW);
    
#ifdef TEST
    if (elemSquareWPixel!=1 && unifyNearW !=0)
        {ERRPRINT("TODO MAKE TEST CASE FOR THIS CONFIG");goto _free;}
    if (checkDenseMatrixToPPM(M,N, (uchar (*)[N]) data->data,
      (double (*)[N]) denseMat,elemSquareWPixel,unifyNearW))   goto _free;
    //TODO add support for step, unifyNearW, NOW CONSIDERED AS dflt... ->1a1 mat
#endif
    out = EXIT_SUCCESS;
    
    _free:
    if(outFd)   close(outFd);
    if(data)    free(data);
    if(map){
        if(munmap(map,dataLen) == -1){
            perror("Error un-mmapping the file");
        }
    //if(ftruncate(outFd,actualLen)<0){perror("ftruncate err ");goto _free;}//remove excess from mmapped
    }
    return out;
}
#endif
