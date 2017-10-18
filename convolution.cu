#include <stdio.h>
#include <helper_timer.h>
#include <helper_cuda.h>
/**
 * in: Input image
 * out: Output image
 * dataSizeX: X dimension of input image
 * dataSizeY: Y dimension of input image
 * kernel: Kernel to use for convolution
 * kernelSizeX: X dimension of kernel
 * kernelSizeY: Y dimension of kernel
 **/


__global__ void convolve2D(float* in, float* out, int dataSizeX, int dataSizeY,
                    float* kernel, int kernelSizeX, int kernelSizeY)
{	
	int ii,jj;		//indexes to check boundaries
	int mm;			//inverse y parameter for traversing the kernel
	int nn;			//inverse x parameter for traversing the kernel
	int threadId = blockIdx.x*blockDim.x+threadIdx.x;	//thread ID on the grid
	float sum = 0;	//variable to hold temporal values
	
	//i will traverse the image as the image is in a 1D array,
	//we will start on our current thread
	//and will advance +=blocks*threads
	//j will traverse the image Y axis
	//ki will traverse the kernel X axis
	//kj will traverse the kernel Y axis
	for(int i=threadId;i<dataSizeX*dataSizeY;i+=gridDim.x*blockDim.x)
	{
		printf("current value we are working on: %f, index: %d\n", in[i],i);
		sum=0;
		for(int kj=0;kj<kernelSizeY;kj++){
			mm = kernelSizeY - 1 - kj;											//index flip for kernel j
			for(int ki=0;ki<kernelSizeX;ki++){
				nn = kernelSizeX - 1 - ki;										//index flip for kernel x
				ii = (i%dataSizeX) + ki-kernelSizeX/2;							//checking if we are out of bounds
				jj = i/dataSizeX+(kj-kernelSizeY/2);							//checking if we are out of bounds
				//we will ignore those ii and jj that are out of bounds
				if(ii>=0 && ii < dataSizeX && jj>=0 && jj < dataSizeY){
					//printf("will add %f to sum from values input[%d,%d]*kernel[%d,%d]: %f*%f \n",
					//	in[ii+jj*dataSizeX] * kernel[nn+mm*kernelSizeX], ii,jj,nn,mm,in[ii+jj*dataSizeX],kernel[nn+mm*kernelSizeX]);
					sum += in[ii+jj*dataSizeX] * kernel[nn+mm*kernelSizeX];		//if we are not out of bounds, we will add the result to sum
				}
			}
		}
		//printf("sum is: %f, out index is: %d\n", sum, i);
		//printf("--------------------------------------------------------------------------------------------------------\n");
		out[i] = sum;															//after performing all calculations for this specific point, we write to the output matrix
	} 
}



int main(){
	//cuda parameters
	int threads = 1;
	int blocks = 1;
	
	printf("Starting the algorithm\n");
	
	StopWatchInterface *timer = NULL;				//variable to hold timer
	
	int imageSizeX = 4;
	int imageSizeY = 3;
	
	int kernelSizeX = 3;
	int kernelSizeY = 3;
	
	int dataSize = sizeof(float)*imageSizeX*imageSizeY;
	
	float* IentryImage = (float*)malloc(dataSize);		//creating an array that will be the input image
	
	float* entryImage;								//image usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&entryImage, dataSize) );
	
	float* IoutputImage = (float*)malloc(dataSize);		//creating an array that will be the output image
	
	float* outputImage;								//image usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&outputImage, dataSize) );
	
	float* Ikernel = (float*)malloc(dataSize);			//creating an array that will be the kernel
	
	float* kernel;									//kernel usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&kernel, kernelSizeX*kernelSizeY*sizeof(float)) );
	
	//generating data to use for image.. imageSizeX*imageSizeY "image"
	//for(int j=0;j<imageSizeY;j++)
	//	for(int i=0;i<imageSizeY;i++)
	//		IentryImage[i+j*imageSizeX] = i+j*imageSizeX+1;
	
	IentryImage[0] = 1;
	IentryImage[1] = 5;
	IentryImage[2] = 2;
	IentryImage[3] = 3;
	IentryImage[4] = 8;
	IentryImage[5] = 7;
	IentryImage[6] = 3;
	IentryImage[7] = 6;
	IentryImage[8] = 3;
	IentryImage[9] = 3;
	IentryImage[10] = 9;
	IentryImage[11] = 1;
	
	//will now generate a kernelSizeX*kernelSizeY kernel
	//for(int j=0;j<kernelSizeX;j++)
	//	for(int i=0;i<kernelSizeY;i++)
	//		Ikernel[i+j*kernelSizeX] = 1;
	Ikernel[0] = 1;
	Ikernel[1] = 2;
	Ikernel[2] = 3;
	Ikernel[3] = 0;
	Ikernel[4] = 0;
	Ikernel[5] = 0;
	Ikernel[6] = 6;
	Ikernel[7] = 5;
	Ikernel[8] = 4;

	//copying the kernel to the GPU memory
	//CheckCudaErrors( cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE) );
	//copying the data to the CPU
	checkCudaErrors(cudaMemcpy(entryImage, IentryImage, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(outputImage, IoutputImage, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(kernel, Ikernel, kernelSizeX*kernelSizeY*sizeof(float), cudaMemcpyHostToDevice));


	printf("Starting the timer\n");
	//we create the timer
	sdkCreateTimer(&timer);
	//we start the timer
	sdkStartTimer(&timer);
	//will now start the 2D convolution process now that we have data
	convolve2D<<<blocks,threads>>>(entryImage,outputImage,imageSizeX,imageSizeY,kernel,kernelSizeX,kernelSizeY);
	checkCudaErrors( cudaDeviceSynchronize() );
	//stop the timer
	sdkStopTimer(&timer);
	//we print how much time did it took to run the convolution
	printf("Elapsed time: %f msec\n",sdkGetTimerValue(&timer));
	
	
	//retrieving values from GPU
	checkCudaErrors( cudaMemcpy(IentryImage, entryImage, dataSize, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(IoutputImage, outputImage, dataSize, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(Ikernel, kernel, kernelSizeX*kernelSizeY*sizeof(float), cudaMemcpyDeviceToHost) );
	
	printf("Entry image was:\n");
	for(int i=0;i<imageSizeX*imageSizeY;i++){
		if(i==0||i%imageSizeX!=0)
			printf("%f, ",IentryImage[i]);
		else printf("\n%f, ", IentryImage[i]);
	}
	printf("\n");
	printf("Output image is:\n");
	for(int i=0;i<imageSizeX*imageSizeY;i++){
		if(i==0||i%imageSizeX!=0)
			printf("%f, ",IoutputImage[i]);
		else printf("\n%f, ", IoutputImage[i]);
	}
	printf("\n");
	printf("Kernel was:\n");
	for(int i=0;i<kernelSizeX*kernelSizeY;i++){
		if(i==0||i%kernelSizeX!=0)
			printf("%f, ",Ikernel[i]);
		else printf("\n%f, ", Ikernel[i]);
	}
	printf("\n");
	return 0;
}
