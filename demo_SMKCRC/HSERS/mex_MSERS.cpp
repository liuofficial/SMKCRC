// Entropy Rate Superpixel Segmentation
#include "mex.h"
#include "matrix.h"
#include "MERCLazyGreedy.h"
#include "MERCInputImage.h"
#include "MERCOutputImage.h"

#include "Image.h"
#include "ImageIO.h"
#include <iostream>
#include <string>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) 
{
	double lambda,sigma;
	int nC,kernel = 0;
	int row,col;
	int conn8 = 1;
	double *pLambda,*pSigma,*pNC;	
	double *data;
	double *out;
	MERCLazyGreedy merc;	
	int dist; // new
	double *pDist; // new
	
	if(!(nrhs==5))
	{
		mexErrMsgTxt("[labels] = mex_MSERS(image,nC,lambda,sigma,dist)\n");
	}
	
    /* Check data type of input argument  */
    if (!(mxIsDouble(prhs[0])))
	{
		mexErrMsgTxt("Input argument must be of type double.");
    }		

	data   = mxGetPr(prhs[0]);
	pNC     = mxGetPr(prhs[1]);	
	pLambda = mxGetPr(prhs[2]);
	pSigma  = mxGetPr(prhs[3]);
	pDist   = mxGetPr(prhs[4]);
	lambda  = *pLambda;
	dist    = (int)*pDist;
	
	int nCs_size = (int)mxGetM(prhs[1]) * (int)mxGetN(prhs[1]);
	int sigs_size = (int)mxGetM(prhs[3]) * (int)mxGetN(prhs[3]);
	
	
	// modified ---------------------------------begin----------------------------------
    int nDims = (int)mxGetNumberOfDimensions(prhs[0]);
    int height, width, nPics;
    if (nDims == 3)
    {
        const int *dim_array;
        dim_array = (int*)mxGetDimensions(prhs[0]);
        height = *dim_array;
        width = *(dim_array+1);
        nPics = *(dim_array+2);
    }else
    {
        width = mxGetN(prhs[0]);
		height = mxGetM(prhs[0]);
        nPics = 1;
    }
	Image<HSMap> inputImage;
	// Create Iamge
	inputImage.Resize(width,height,true);
	// Read the image from MATLAB
	for (col=0; col < width; col++)
	{
		for (row=0; row < height; row++)
		{
			int *data = new int[nPics];
			for (int i=0; i<nPics; i++)
				data[i] = (int)mxGetPr(prhs[0])[row+col*height+i*width*height];
			HSMap spec(data, nPics);
			delete [] data;
			data = NULL;
			inputImage.Access(col,row) = spec;	
		}
	}
	// Allocate memory for the labeled image.
	plhs[0] = mxCreateDoubleMatrix(height*width, nCs_size*sigs_size, mxREAL);
	out =  mxGetPr(plhs[0]);
	for (int i=0; i<nCs_size; i++)
	{
		nC = (int)pNC[i];
		for (int j=0; j<sigs_size; j++)
		{
			sigma = pSigma[j];
			MERCInputImage<HSMap> input;
			// Read the image for segmentation
	        input.Read_Image(&inputImage,conn8, dist);
			// Entropy rate superpixel segmentation
			MERCLazyGreedy merclg;
			merclg.ClusteringTreeIF(input.nNodes_,input,kernel,sigma,lambda*1.0*nC,nC);
			vector<int> label = MERCOutputImage::DisjointSetToLabel(merclg.disjointSet_);
			// Fill in the labeled image
			for (col=0; col < width; col++)
				for (row=0; row < height; row++)
					out[row+col*height + j*width*height + i*width*height*sigs_size] = (double)label[col+row*width];	
		}
	}
	// modified ---------------------------------end----------------------------------
	return;
}