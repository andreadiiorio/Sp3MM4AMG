#ifndef SPMAT_IMG
#define SPMAT_IMG


#define PPM_HEADER_MAX_LEN 100
#define PPM_CHAN_NUM       3
#define RGB_BLACK          0xFFFFFF

#define PATTERN_DENSE      "PATTERN_DENSE"
#define MM_COO             "MM_COO"
#define DFLT_OUTPATH       "/tmp/mat.ppm"
typedef struct{
    char* data;     //width*height*3 triples of RGB pixel
    ulong width;

    ulong height;
    char header[PPM_HEADER_MAX_LEN];
}   ppmData;

#define NNZ_PIXEL_COLOR  255
#define Z_PIXEL_COLOR    0
/*
 * convert dense matrix @mat into ppm RGB triple pixels ( in @data ) 
 * with a black dot per NZ elem 
 * assign each consecutive @step^2 matrix elements to a dot in the PPM image
 * if at least 1 nz is in the square 
 * the dot will also include unifyNearW^2 image pixel to let the dot be more visible
 */
void denseMatrixToPPM(ppmData* data,
  uint M, uint N, double mat[][N],ushort step, ushort unifyNearW);


/*
 * map each nz eleme in @sparseMat into a pixel dots square
 * similarly as before in _denseMatrixToPPM
 */
void sparseMatrixToPPM(ppmData* data,spmat* sparseMat,ushort step, ushort unifyNearW);

#endif
