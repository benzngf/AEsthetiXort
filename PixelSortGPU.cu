#include "PixelSort.h"
#include <cuda.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/*__global__ void  PixelSortGPULinear(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
(PixelSortPatternParmLinear *)pattern_parm, anti_aliasing)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
}*/
/*
__global__ void CalculateFixed(
	const float *background, const float *target, const float* mask, float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

}
*/
void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

	switch (pattern_parm->pattern) {
	case PSP_Linear:
		/*PixelSortGPULinear(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
			(PixelSortPatternParmLinear *)pattern_parm, anti_aliasing);*/
		break;
		/* case PSP_Radial_Zoom:
		PixelSortGPURadialZoom(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
		(PixelSortPatternParmRadialZoom *)pattern_parm, anti_aliasing);*/
		/*
		case PSP_Radial_Spin:
		case PSP_Polygon:
		case PSP_Spiral:
		case PSP_Sine:
		case PSP_Triangle:
		case PSP_Saw_Tooth:
		case PSP_Optical_Flow:
		*/

	default:
		break;
	}
}



///////////////////////CODE TO DETECT CUDA/////////////////////////////////
/*
for CUDA supporting...

After Effects does not support CUDA based code, if the following functions are there in CUDA programe.
Please remember dont use the below functions while writing the CUDA code.
1) printf()
2) puts()
3) getchar()
4) exit()
5) All timer related functions like cutCreateTimer()
6) dont use CUDA_SAFE_CALL and CUT_SAFE_CALL macros ( which are mostly used in CUDA SDK samples ).
7) CUT_DEVICE_INIT and CUT_EXIT macros
*/

extern "C"
int callCudaFunc();

__global__
void Kernal(int a, int b, int* sum)
{
	*sum = a + b;
}

extern "C"
int callCudaFunc()
{
	// initialise Device
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount < 0)
		return 0;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceCount);

	// Allocate memory on Device
	int* D_sum = NULL;
	cudaError error = cudaMalloc((void**)&D_sum, sizeof(int) * 1);


	// global function
	Kernal << <1, 1 >> >(1, 2, D_sum);


	// copy data from Device memory to host memory
	int H_sum = 0;
	error = cudaMemcpy(&H_sum, D_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	// return 
	return H_sum;
}