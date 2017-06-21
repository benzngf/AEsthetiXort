#include "PixelSort.h"

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/*__global__ void  PixelSortGPULinear(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
(PixelSortPatternParmLinear *)pattern_parm, anti_aliasing)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
}*/
__global__ void CalculateFixed(
	const float *background, const float *target, const float* mask, float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

}

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing) {

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