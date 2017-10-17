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


__global__ void convolve2D(int* in, int* out, int dataSizeX, int dataSizeY,
                    int* kernel, int kernelSizeX, int kernelSizeY)
{	
	int ii,jj;		//indexes to check boundaries
	int mm;			//inverse y parameter for traversing the kernel
	int nn;			//inverse x parameter for traversing the kernel
	int threadId = blockIdx.x*blockDim.x+threadIdx.x;	//thread ID on the grid
	int sum = 0;	//variable to hold temporal values
	printf("Thread id: %d\n",threadId);
	
	//i will traverse the image as the image is in a 1D array,
	//we will start on our current thread
	//and will advance +=blocks*threads
	//j will traverse the image Y axis
	//ki will traverse the kernel X axis
	//kj will traverse the kernel Y axis
	//printf("gridDim.x*blockDim.x: %d\n",gridDim.x*blockDim.x);
	
	for(int i=threadId;i<dataSizeX*dataSizeY;i+=gridDim.x*blockDim.x)
	{
		printf("current value we are working on: %d, index: %d\n", in[i],i);
		int currI = i%dataSizeX;
		int currJ = i/dataSizeX;
		//printf("current i: %d and current j: %d, current value we are \
			//working on: %d\n",currI, currJ), in[currI+currJ*dataSizeX];
		printf("currI+currJ*dataSizeX: %d\n", currI+currJ*dataSizeX);
		//printf("test i: %d, im thread number %d\n",i,threadId);
		sum=0;
		for(int kj=0;kj<kernelSizeY;kj++){
			
			mm = kernelSizeY - 1 - kj;
			
			printf("flipped j value for kernel: %d\n",mm);
			printf("kernel size X is: %d\n",kernelSizeX);
			for(int ki=0;ki<kernelSizeX;ki++){
				nn = kernelSizeX - 1 - ki;
				printf("flipped x value for kernel: %d\n",nn);
				printf("current j: %d\n",i/dataSizeX);
				
				ii = (i%dataSizeX) + ki-kernelSizeX/2;
				printf("ii = %d\n",ii);
				//ii = threadIdx.x+(kj-kernelSizeX/2);
				jj = i/dataSizeX+(kj-kernelSizeY/2);
				
				printf("jj = %d\n",jj);
				printf("ii+jj*dataSizeX: %d\n",ii+jj*dataSizeX);
				//we will ignore those ii and jj that are out of bounds
				if(ii>=0 && ii < dataSizeX && jj>=0 && jj < dataSizeY){
					printf("will add %d to sum from values input[%d,%d]*kernel[%d,%d]: %d*%d \n",
						in[ii+jj*dataSizeX] * kernel[nn+mm*kernelSizeX], ii,jj,nn,mm,in[ii+jj*dataSizeX],kernel[nn+mm*kernelSizeX]);
					sum += in[ii+jj*dataSizeX] * kernel[nn+mm*kernelSizeX];
				}
					//sum+=1;
			}
		}
		printf("sum is: %d, out index is: %d\n", sum, i);
		printf("/////////////////////////////\n");
		out[i] = sum;
	} 
}



int main(){
	//cuda parameters
	int threads = 1;
	int blocks = 1;
	
	printf("Starting the algorithm\n");
	
	StopWatchInterface *timer = NULL;				//variable to hold timer
	
	int imageSizeX = 3;
	int imageSizeY = 3;
	
	int kernelSizeX = 3;
	int kernelSizeY = 3;
	
	int dataSize = sizeof(int)*imageSizeX*imageSizeY;
	
	int* IentryImage = (int*)malloc(dataSize);		//creating an array that will be the input image
	
	int* entryImage;								//image usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&entryImage, dataSize) );
	
	int* IoutputImage = (int*)malloc(dataSize);		//creating an array that will be the output image
	
	int* outputImage;								//image usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&outputImage, dataSize) );
	
	int* Ikernel = (int*)malloc(dataSize);			//creating an array that will be the kernel
	
	int* kernel;									//kernel usable by cuda
	checkCudaErrors( cudaMalloc( (void **)&kernel, kernelSizeX*kernelSizeY*sizeof(int)) );
	
	//generating data to use for image.. imageSizeX*imageSizeY "image"
	for(int j=0;j<imageSizeY;j++)
		for(int i=0;i<imageSizeY;i++)
			IentryImage[i+j*imageSizeX] = i+j*imageSizeX+1;
	
	//IentryImage[0] = 1;
	//IentryImage[1] = 5;
	//IentryImage[2] = 2;
	//IentryImage[3] = 3;
	//IentryImage[4] = 8;
	//IentryImage[5] = 7;
	//IentryImage[6] = 3;
	//IentryImage[7] = 6;
	//IentryImage[8] = 3;
	//IentryImage[9] = 3;
	//IentryImage[10] = 9;
	//IentryImage[11] = 1;
	
	//will now generate a kernelSizeX*kernelSizeY kernel
	//for(int j=0;j<kernelSizeX;j++)
	//	for(int i=0;i<kernelSizeY;i++)
	//		Ikernel[i+j*kernelSizeX] = i+j*kernelSizeX;
	Ikernel[0] = -1;
	Ikernel[1] = -2;
	Ikernel[2] = -1;
	Ikernel[3] = 0;
	Ikernel[4] = 0;
	Ikernel[5] = 0;
	Ikernel[6] = 1;
	Ikernel[7] = 2;
	Ikernel[8] = 1;

	//copying the kernel to the GPU memory
	//heckCudaErrors( cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE) );
	//copying the data to the CPU
	checkCudaErrors(cudaMemcpy(entryImage, IentryImage, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(outputImage, IoutputImage, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(kernel, Ikernel, kernelSizeX*kernelSizeY*sizeof(int), cudaMemcpyHostToDevice));


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
	checkCudaErrors( cudaMemcpy(Ikernel, kernel, kernelSizeX*kernelSizeY*sizeof(int), cudaMemcpyDeviceToHost) );
	
	printf("Entry image was:\n");
	for(int i=0;i<imageSizeX*imageSizeY;i++){
		if(i==0||i%imageSizeX!=0)
			printf("%d, ",IentryImage[i]);
		else printf("\n%d, ", IentryImage[i]);
	}
	printf("\n");
	printf("Output image is:\n");
	for(int i=0;i<imageSizeX*imageSizeY;i++){
		if(i==0||i%imageSizeX!=0)
			printf("%d, ",IoutputImage[i]);
		else printf("\n%d, ", IoutputImage[i]);
	}
	printf("\n");
	printf("Kernel was:\n");
	for(int i=0;i<kernelSizeX*kernelSizeY;i++){
		if(i==0||i%kernelSizeX!=0)
			printf("%d, ",Ikernel[i]);
		else printf("\n%d, ", Ikernel[i]);
	}
	printf("\n");
	return 0;
}
