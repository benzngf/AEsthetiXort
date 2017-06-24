#include "PixelSort.h"
#include <stdio.h>
#include <math.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ __host__ int MaxRGB(const int R, const int G, const int B) {
    if(R > G && R > B) return R;
    else if (G > R && G > B) return G;
    else if (B > R && B > G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ int MinRGB(const int R, const int G, const int B) {
    if(R < G && R < B) return R;
    else if (G < R && G < B) return G;
    else if (B < R && B < G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ int absolute(int x){
    if (x >= 0)
        return x;
    return -x;
}
__device__ __host__ int getLuminance(const int R, const int G, const int B) {
    int M = MaxRGB(R, G, B);
    int m = MinRGB(R, G, B);
    return (M + m) / 2;
}  
__device__ __host__ int getHue(const int R, const int G, const int B) {
    int M = MaxRGB(R, G, B);
    int m = MinRGB(R, G, B);
    int C = M - m;

    if (C == 0)
        return 0;
    else if (M == R)
        return 60 * ( ( (G - B) / C ) % 6 );
    else if (M == G)
        return 2 + ( (B - R) / C );
    else if (M == B)
        return 4 + ( (R - G) / C );
    return 0;
}
__device__ __host__ int getSaturation(const int R, const int G, const int B) {
    int M = MaxRGB(R, G, B);
    int m = MinRGB(R, G, B);
    int C = M - m;
    int L = (M + m) / 2;

    if (L == 1)
        return 0;
    else
        return C / (1 - absolute(2 * L - 1));
}


#ifdef DEBUG
#define debug_print(...) fprintf(stderr, __VA_ARGS__) 
#else
#define debug_print(...)
#endif

#define OUPUT_POINT_MAX 1000

//#define PREDEBUG

// TODO: I think these code is GPU-unfriendly
// This shoulb be __device__
#ifdef PREDEBUG
__global__  void GetListToSort(
#else
__device__  void GetListToSort(
#endif
        PixelSortPatternParmLinear *linear, 
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, float *output) {
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

    *order = cnt-1;

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


#ifndef PREDEBUG
__global__ void SortFromList(PixelSortPatternParmLinear *linear, 
    const Pixel *input, Pixel *output, 
    const int w, const int h,
    const PixelSortBy sort_by) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelid = y * w + x;

    if (x < w and y < h) {
        // Get a list for sorting
        const float pixelx = x + 0.5;
        const float pixely = y + 0.5;

        int point_cnt_gpu;
        int order_gpu;
        float sort_list_gpu[2*OUPUT_POINT_MAX];


        GetListToSort(linear, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, sort_list_gpu);

        // Sorting: preprocessing
        int converted_sort_list_gpu[OUPUT_POINT_MAX];
        Pixel pixel_list_gpu[OUPUT_POINT_MAX];
        // Fill sorting pixel
        for (int i = 0; i < point_cnt_gpu; i++){
            int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
            pixel_list_gpu[i].r = input[ind].r;
            pixel_list_gpu[i].g = input[ind].g;
            pixel_list_gpu[i].b = input[ind].b;
            pixel_list_gpu[i].a = input[ind].a;

        }
        // Fill sorting key
        switch (sort_by){

            case PSB_R:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = input[ind].r;
                }
                break;
            case PSB_G:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = input[ind].g;
                }
                break;
            case PSB_B:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = input[ind].b;
                }
                break;
            case PSB_Hue:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = getHue(input[ind].r, input[ind].g, input[ind].b);
                }
                break;
            case PSB_Saturation:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = getSaturation(input[ind].r, input[ind].g, input[ind].b);
                }
                break;
            case PSB_Luminance:
                for (int i = 0; i < point_cnt_gpu; i++) {
                    int ind = (int)floorf(sort_list_gpu[2*i+1])*w + (int)floorf(sort_list_gpu[2*i]);
                    converted_sort_list_gpu[i] = getLuminance(input[ind].r, input[ind].g, input[ind].b);
                }
                break;
            default:
                break;
        }

        // Sort
        thrust::stable_sort_by_key(thrust::device, converted_sort_list_gpu, converted_sort_list_gpu + point_cnt_gpu, pixel_list_gpu);

        // Fill value: order

        output[pixelid].r = pixel_list_gpu[order_gpu].r;
        output[pixelid].g = pixel_list_gpu[order_gpu].g;
        output[pixelid].b = pixel_list_gpu[order_gpu].b;
        output[pixelid].a = pixel_list_gpu[order_gpu].a;



    }



}
#endif
/*
__global__ sort() {
    const int x = ;
    const int y = ;
    const int pixelid = y*w + x;
    
    if (x and y are in the valid location) {
        const float pixelx = x + 0.5, pixely = y + 0.5;

        int order, point_cnt;
        float output[OUPUT_POINT_MAX*2];

        GetListToSort( PixelSortPatternParmLinear *linear, pixelx, pixely, w,  h, &order, &point_cnt, &output);

           //point -> SortBy


        thrust::sort(output);

        background[pixelid] = output[order];

    }
}
*/
/*(input image, image width, image height, output image(to fill),
sort by? (RGB...), threshold_min, threshold, max, reverse?
pattern parameter, do antialiasing?, sort alpha?)*/

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

    PixelSortPatternParm *pattern_parm_gpu = nullptr;
	switch (pattern_parm->pattern) {
        case PSP_Linear:
            {
            debug_print("PSP_Linear (%d)\n", pattern_parm->pattern);
            debug_print("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmLinear));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmLinear), cudaMemcpyHostToDevice);
#ifndef PREDEBUG            
            dim3 gdim(CeilDiv(width, 32), CeilDiv(height, 16)), bdim(32, 16);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, 
                                            input, output, width, height, sort_by);
#endif            
            break;
            }
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
#ifdef PREDEBUG    
    {
        int *point_cnt = (int *)malloc(sizeof(int));
        int *point_cnt_gpu;

        int *order = (int *)malloc(sizeof(int));
        int *order_gpu;

        float *sort_list = (float *)malloc(sizeof(float)*OUPUT_POINT_MAX*2);
        float *sort_list_gpu;

        cudaMalloc(&sort_list_gpu, sizeof(float) * OUPUT_POINT_MAX*2);
        cudaMalloc(&point_cnt_gpu, sizeof(int));
        cudaMalloc(&order_gpu, sizeof(int));

        debug_print("w:%d, h:%d\n", width, height);
        GetListToSort<<<1, 1>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, 500.0f, 300.0f, (float)width, (float)height, order_gpu, point_cnt_gpu, sort_list_gpu);

        cudaMemcpy(point_cnt, point_cnt_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(order, order_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sort_list, sort_list_gpu, sizeof(float)*2*(*point_cnt), cudaMemcpyDeviceToHost);

        printf("%d, %d\n", *order, *point_cnt);
        for (int i = 0; i < *point_cnt; ++i) {
            printf("%d: (%f, %f)\n", i, sort_list[2*i], sort_list[2*i+1]);
        }

        free(sort_list);
        cudaFree(sort_list_gpu);
        free(order);
        cudaFree(order_gpu);
        free(point_cnt);
        cudaFree(point_cnt_gpu);
    }
#endif
    

    //cudaMemcpy(output, input, width*height*sizeof(Pixel), cudaMemcpyDeviceToDevice);
    cudaFree(pattern_parm_gpu);
}
