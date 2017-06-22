#include "PixelSort.h"
#include <stdio.h>

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
            printf("PSP_Linear (%d)\n", pattern_parm->pattern);
            printf("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            break;
        case PSP_Radial_Spin:
            printf("PSP_Radial_Spin (%d)\n", pattern_parm->pattern);
            printf("center: (%f, %f)\n", 
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[0],
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[1]);
            printf("WHRatio: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->WHRatio);
            printf("rotation: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->rotation);
            break;
        case PSP_Polygon:
            printf("PSP_Polygon (%d)\n", pattern_parm->pattern);
            printf("center: (%f, %f)\n", 
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[0],
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[1]);
            printf("numSides: %d\n", ((PixelSortPatternParmPolygon *)pattern_parm)->numSides);
            printf("WHRatio: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->WHRatio);
            printf("rotation: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->rotation);
            break;
        case PSP_Spiral:
            printf("PSP_Spiral (%d)\n", pattern_parm->pattern);
            printf("center: (%f, %f)\n", 
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[0],
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[1]);
            printf("curveAngle: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->curveAngle);
            printf("WHRatio: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->WHRatio);
            printf("rotation: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->rotation);
            break;
        case PSP_Sine:
            printf("PSP_Sine (%d)\n", pattern_parm->pattern);
            printf("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            printf("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            printf("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            break;
        case PSP_Triangle:
            printf("PSP_Triangle (%d)\n", pattern_parm->pattern);
            printf("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            printf("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            printf("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            break;
        case PSP_Saw_Tooth:
            printf("PSP_Saw_Tooth (%d)\n", pattern_parm->pattern);
            printf("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            printf("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            printf("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            break;
        case PSP_Optical_Flow:
            printf("PSP_Optical_Flow (%d)\n", pattern_parm->pattern);
            break;

        default:
            break;
	}
    cudaMemcpy(output, input, width*height*sizeof(Pixel), cudaMemcpyDeviceToDevice);
}
