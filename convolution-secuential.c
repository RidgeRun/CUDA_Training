#include <stdio.h>
#include <getopt.h>
#include <time.h>
#include <stdlib.h>
/**
 * in: Input image
 * out: Output image
 * dataSizeX: X dimension of input image
 * dataSizeY: Y dimension of input image
 * kernel: Kernel to use for convolution
 * kernelSizeX: X dimension of kernel
 * kernelSizeY: Y dimension of kernel
 **/


void convolve2D(float* in, float* out, int dataSizeX, int dataSizeY,
                    float* kernel, int kernelSizeX, int kernelSizeY)
{	
	int ii,jj;		//indexes to check boundaries
	int mm;			//inverse y parameter for traversing the kernel
	int nn;			//inverse x parameter for traversing the kernel
	float sum = 0;	//variable to hold temporal values
	
	//i will traverse the image as the image is in a 1D array,
	//we will start on our current thread
	//and will advance +=blocks*threads
	//j will traverse the image Y axis
	//ki will traverse the kernel X axis
	//kj will traverse the kernel Y axis
	for(int i=0;i<dataSizeX*dataSizeY;i++)
	{
		//printf("current value we are working on: %f, index: %d\n", in[i],i);
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



int main(int argc, char **argv){
	
	int c;
  int debugFlag = 0;
  
  
  int imageSizeX = 0;
  int imageSizeY = 0;
  int kernelSizeX = 0;
  int kernelSizeY = 0;
  while (1)
    {
      static struct option long_options[] =
        {
          /* These options don’t set a flag.
             We distinguish them by their indices. */
          {"debug",     no_argument,       0, 'd'},
          {"imageSizeX",    required_argument, 0, 'x'},
          {"imageSizeY",    required_argument, 0, 'y'},
          {"kernelSizeX",    required_argument, 0, 'i'},
          {"kernelSizeY",    required_argument, 0, 'j'},
          {0, 0, 0, 0}
        };
      /* getopt_long stores the option index here. */
      int option_index = 0;

      c = getopt_long (argc, argv, "dx:y:i:j:",
                       long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
        break;

      switch (c)
        {
        case 0:
          /* If this option set a flag, do nothing else now. */
          if (long_options[option_index].flag != 0)
            break;
          printf ("option %s", long_options[option_index].name);
          if (optarg)
            printf (" with arg %s", optarg);
          printf ("\n");
          break;

        case 'd':
		  debugFlag = 1;
          //puts ("option -d\n");
          break;

        case 'x':
          imageSizeX = atoi(optarg);
          //printf ("option -x with value `%s'\n", optarg);
          break;

        case 'y':
          imageSizeY = atoi(optarg);
          //printf ("option -y with value `%s'\n", optarg);
          break;
		case 'i':
		  kernelSizeX = atoi(optarg);
          //printf ("option -i with value `%s'\n", optarg);
          break;

        case 'j':
          kernelSizeY = atoi(optarg);
          //printf ("option -j with value `%s'\n", optarg);
          break;
        case '?':
          /* getopt_long already printed an error message. */
          break;

        default:
          abort ();
        }
    }
    //if no value were given for image and kernel sizes, give the default value
    imageSizeX = (imageSizeX==0?256:imageSizeX);
	imageSizeY = (imageSizeY==0?256:imageSizeY);
	kernelSizeX = (kernelSizeX==0?5:kernelSizeX);
	kernelSizeY = (kernelSizeY==0?5:kernelSizeY);
	
	
	
	printf("Starting the algorithm\n");
	
	
	int dataSize = sizeof(float)*imageSizeX*imageSizeY;
	
	float* entryImage = (float*)malloc(dataSize);		//creating an array that will be the input image
	
	float* outputImage = (float*)malloc(dataSize);		//creating an array that will be the output image
	
	float* kernel = (float*)malloc(dataSize);			//creating an array that will be the kernel
	
	
	//generating data to use for image.. imageSizeX*imageSizeY "image" populated with its respecting i value
	for(int j=0;j<imageSizeY;j++)
		for(int i=0;i<imageSizeX;i++)
			entryImage[i+j*imageSizeX] = i+j*imageSizeX+1;
	
	
	//will now generate a kernelSizeX*kernelSizeY kernel populated with 1's
	for(int j=0;j<kernelSizeY;j++)
		for(int i=0;i<kernelSizeX;i++)
			kernel[i+j*kernelSizeX] = 1;


	printf("Starting the timer\n");
	//we start the timer
	clock_t start, end;
	start = clock();
	//will now start the 2D convolution process now that we have data
	convolve2D(entryImage,outputImage,imageSizeX,imageSizeY,kernel,kernelSizeX,kernelSizeY);
	//stop the timer
	end = clock();
	//we print how much time did it took to run the convolution
	float diff = ((float)(end - start) / 1000000.0F ) * 1000;   
	
	printf("Elapsed time: %f msec\n",diff);
	
	if(debugFlag){
		printf("Entry image was:\n");
		for(int i=0;i<imageSizeX*imageSizeY;i++){
			if(i==0||i%imageSizeX!=0)
				printf("%f, ",entryImage[i]);
			else printf("\n%f, ", entryImage[i]);
		}
		printf("\n");
		printf("Output image is:\n");
		for(int i=0;i<imageSizeX*imageSizeY;i++){
			if(i==0||i%imageSizeX!=0)
				printf("%f, ",outputImage[i]);
			else printf("\n%f, ", outputImage[i]);
		}
		printf("\n");
		printf("Kernel was:\n");
		for(int i=0;i<kernelSizeX*kernelSizeY;i++){
			if(i==0||i%kernelSizeX!=0)
				printf("%f, ",kernel[i]);
			else printf("\n%f, ", kernel[i]);
		}
		printf("\n");
	}
	return 0;
}
