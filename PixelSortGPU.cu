#include "PixelSort.h"
#include <stdio.h>
#include <math.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#ifdef DEBUG
#define debug_print(...) fprintf(stderr, __VA_ARGS__) 
#else
#define debug_print(...)
#endif

#define OUPUT_POINT_MAX 1000

// TODO: I think these code is GPU-unfriendly
__global__  void GetListToSort(PixelSortPatternParmLinear *linear, const float x, const float y, const float w, const float h, int *point_cnt, float *output) {
    float delta[2], last[2];
    int cnt = 1;

    delta[0] = cos(linear->angle);
    delta[1] = sin(linear->angle);

    output[0] = x;
    output[1] = y;

    // prev
    last[0] = x - delta[0];
    last[1] = y - delta[1];
    while (cnt < OUPUT_POINT_MAX && last[0] > 0 && last[0] < w && last[1] > 0 && last[1] < h) {
        output[cnt*2] = last[0];
        output[cnt*2+1] = last[1];
        ++cnt;
        last[0] -= delta[0];
        last[1] -= delta[1];
    }

    // next
    last[0] = x + delta[0];
    last[1] = y + delta[1];
    while (cnt < OUPUT_POINT_MAX && last[0] > 0 && last[0] < w && last[1] > 0 && last[1] < h) {
        output[cnt*2] = last[0];
        output[cnt*2+1] = last[1];
        ++cnt;
        last[0] += delta[0];
        last[1] += delta[1];
    }

    *point_cnt = cnt;
}

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

    PixelSortPatternParm *pattern_parm_gpu = nullptr;
	switch (pattern_parm->pattern) {
        case PSP_Linear:
            debug_print("PSP_Linear (%d)\n", pattern_parm->pattern);
            debug_print("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmLinear));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmLinear), cudaMemcpyHostToDevice);
            break;
        case PSP_Radial_Spin:
            debug_print("PSP_Radial_Spin (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[0],
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[1]);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmRadialSpin));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmRadialSpin), cudaMemcpyHostToDevice);
            break;
        case PSP_Polygon:
            debug_print("PSP_Polygon (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[0],
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[1]);
            debug_print("numSides: %d\n", ((PixelSortPatternParmPolygon *)pattern_parm)->numSides);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmPolygon));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmPolygon), cudaMemcpyHostToDevice);
            break;
        case PSP_Spiral:
            debug_print("PSP_Spiral (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[0],
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[1]);
            debug_print("curveAngle: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->curveAngle);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmSpiral));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmSpiral), cudaMemcpyHostToDevice);
            break;
        case PSP_Sine: 
        case PSP_Triangle: 
        case PSP_Saw_Tooth:
            debug_print("PSP_Sine (%d)\n", pattern_parm->pattern);
            debug_print("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            debug_print("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            debug_print("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmWave));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmWave), cudaMemcpyHostToDevice);
            break;
        case PSP_Optical_Flow:
            debug_print("PSP_Optical_Flow (%d)\n", pattern_parm->pattern);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmOpFlow));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmOpFlow), cudaMemcpyHostToDevice);
            break;

        default:
            break;
	}

    // Test code
    {
        int *point_cnt = (int *)malloc(sizeof(int));
        int *point_cnt_gpu;

        float *sort_list = (float *)malloc(sizeof(float)*OUPUT_POINT_MAX*2);
        float *sort_list_gpu;

        cudaMalloc(&sort_list_gpu, sizeof(float) * OUPUT_POINT_MAX*2);
        cudaMalloc(&point_cnt_gpu, sizeof(int));

        debug_print("w:%d, h:%d\n", width, height);
        GetListToSort<<<1, 1>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, 500.0f, 300.0f, (float)width, (float)height, point_cnt_gpu, sort_list_gpu);

        cudaMemcpy(point_cnt, point_cnt_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sort_list, sort_list_gpu, sizeof(float)*2*(*point_cnt), cudaMemcpyDeviceToHost);

        printf("%d\n", *point_cnt);
        for (int i = 0; i < *point_cnt; ++i) {
            printf("%d: (%f, %f)\n", i, sort_list[2*i], sort_list[2*i+1]);
        }

        free(sort_list);
        cudaFree(sort_list_gpu);
        free(point_cnt);
        cudaFree(point_cnt_gpu);
    }
    

    cudaMemcpy(output, input, width*height*sizeof(Pixel), cudaMemcpyDeviceToDevice);
    cudaFree(pattern_parm_gpu);
}
