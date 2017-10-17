# Add source files here
EXECUTABLE	:= convolution
# CUDA source files (compiled with cudacc)
CUFILES		:= convolution.cu
# CUDA dependency files
CU_DEPS		:=
# C/C++ source files (compiled with gcc / c++)
# CCFILES		:= convolutionSeparable_gold.cpp \

NVCC = /usr/local/cuda/bin/nvcc
################################################################################
# Rules and targets
#

convolution: $(CUFILES) 
	$(NVCC) $(CUFILES) -o $(EXECUTABLE) -I/usr/local/cuda-8.0/samples/common/inc
