#include "PixelSort.h"
#include <cuda.h>
#ifdef DEBUG
	#include <stdio.h>
	#include <math.h>
#endif
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ __host__ float MaxRGB(const float R, const float G, const float B) {
    if(R > G && R > B) return R;
    else if (G > R && G > B) return G;
    else if (B > R && B > G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ float MinRGB(const float R, const float G, const float B) {
    if(R < G && R < B) return R;
    else if (G < R && G < B) return G;
    else if (B < R && B < G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ float absolute(float x){
    if (x >= 0)
        return x;
    return -x;
}
__device__ __host__ float getLuminance(const float R, const float G, const float B) {
    float R_ = R/255.0f;
    float G_ = G/255.0f;
    float B_ = B/255.0f;
    float M = MaxRGB(R_, G_, B_);
    float m = MinRGB(R_, G_, B_);
    return (M + m) / 2;
}  
__device__ __host__ float getHue(const float R, const float G, const float B) {
    float M = MaxRGB(R, G, B);
    float m = MinRGB(R, G, B);
    float C = M - m;

    float Result = 0.0f;
    if (C > -0.1f && C < 0.1f) //C == 0.0f
        Result = 0.0f;
    else if (M == R)
        Result = fmodf((G - B) / C, 6);
    else if (M == G)
        Result = 2.0f + ( (B - R) / C );
    else if (M == B)
        Result = 4.0f + ( (R - G) / C );
    return 60.0f*Result;
}
__device__ __host__ float getSaturation(const float R, const float G, const float B) {
    float R_ = R/255.0f;
    float G_ = G/255.0f;
    float B_ = B/255.0f;

    float M = MaxRGB(R_, G_, B_);
    float m = MinRGB(R_, G_, B_);
    float C = M - m;
    float L = (M + m) / 2;

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
__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmLinear *linear, 
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output) {
    float delta[2], last[2];
    int cnt = 1;
    
    delta[0] = cos(linear->angle);
    delta[1] = sin(linear->angle);
    
#define PIXELXY(x, y) (input[int(x) + int(y)*int(w)])
    output[0] = PIXELXY(x, y);
    
    // prev
    last[0] = x - delta[0];
    last[1] = y - delta[1];
    while (cnt < OUPUT_POINT_MAX && 
           last[0] > 0 && last[0] < w && 
           last[1] > 0 && last[1] < h &&
           PIXELXY(last[0], last[1]).key > 0.0f) {
        // TODO: AA here
        output[cnt] = PIXELXY(last[0], last[1]);
        ++cnt;
        last[0] -= delta[0];
        last[1] -= delta[1];
    }

    *order = cnt-1;

    // next
    last[0] = x + delta[0];
    last[1] = y + delta[1];
    while (cnt < OUPUT_POINT_MAX && 
           last[0] > 0 && last[0] < w && 
           last[1] > 0 && last[1] < h &&
           PIXELXY(last[0], last[1]).key > 0.0f) {
        // TODO: AA here
        output[cnt] = PIXELXY(last[0], last[1]);
        ++cnt;
        last[0] += delta[0];
        last[1] += delta[1];
    }
#undef PIXELXY

    *point_cnt = cnt;
}

/*
   Assume the domain of threshold_min, threshold_max is [0.0, 100.0]
   TODO: this is a naive version (use branches)
*/
__device__ float Map01(
        float x, 
        float min, float max, 
        float threshold_min, float threshold_max) {
    float len = max - min;
    threshold_min = min + (threshold_min/100.0f)*len;
    threshold_max = min + (threshold_max/100.0f)*len;

    if (threshold_max < threshold_min) {
        if (x > threshold_max && x < threshold_min) {
            return -1.0f;
        }
        float rhalf = max - threshold_min;
        float lhalf = threshold_max - min;
        float p = (x > threshold_max)? threshold_min : min;
        float offset = (x > threshold_max)? 0 : rhalf;

        return ((x - p) + offset) / (rhalf+lhalf);

    } else {
        if (x > threshold_max || x < threshold_min) {
            return -1.0f;
        }
        return (x - threshold_min) / (threshold_max - threshold_min);
    }
}

__global__ void ComputeKey(
        const PixelSortBy sort_by, 
        const int w, const int h,
        const float threshold_min, const float threshold_max,
        Pixel *inout) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        const int pixelid = y * w + x;
        Pixel *cur = inout + pixelid;
        if (cur->a == 0.0f) {
            cur->key = -1.0f;
            return;
        }

        float min, max;
        switch (sort_by) {
            case PSB_R:
                cur->key = cur->r;
                min = 0;
                max = 255;
                break;
            case PSB_G:
                cur->key = cur->g;
                min = 0;
                max = 255;
                break;
            case PSB_B:
                cur->key = cur->b;
                min = 0;
                max = 255;
                break;
            case PSB_Hue:
                cur->key = getHue(cur->r, cur->g, cur->b);
                min = 0;
                max = 360;
                break;
            case PSB_Saturation:
                cur->key = getSaturation(cur->r, cur->g, cur->b);
                min = 0;
                max = 1;
                break;
            case PSB_Luminance:
                cur->key = getLuminance(cur->r, cur->g, cur->b);
                min = 0;
                max = 1;
                break;
            default:
                break;
        }

        cur->key = Map01(cur->key, min, max, threshold_min, threshold_max);
    }
}

#ifndef PREDEBUG
__global__ void SortFromList(PixelSortPatternParmLinear *linear, 
    const Pixel *input, Pixel *output, 
    const int w, const int h) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelid = y * w + x;

    if (x < w && y < h) {
        const float pixelx = x + 0.5;
        const float pixely = y + 0.5;

        int point_cnt_gpu;
        int order_gpu;
        Pixel pixel_list_gpu[OUPUT_POINT_MAX];

        // Get a list for sorting
        GetListToSort(input, linear, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, pixel_list_gpu);

        // Sorting
        for (int i = 0; i < point_cnt_gpu; i++) {
            for (int j = i; j < point_cnt_gpu - i; j++){
                if (pixel_list_gpu[j].key > pixel_list_gpu[j+1].key) {
                    Pixel temp = pixel_list_gpu[j];
                    pixel_list_gpu[j] = pixel_list_gpu[j+1];
                    pixel_list_gpu[j+1] = temp;
                }
            }
        }

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

void PixelSortGPU(Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

    dim3 gdim(CeilDiv(width, 32), CeilDiv(height, 16)), bdim(32, 16);

    ComputeKey<<<gdim, bdim>>>(sort_by, width, height, threshold_min, threshold_max, input);

    PixelSortPatternParm *pattern_parm_gpu = nullptr;
	switch (pattern_parm->pattern) {
        case PSP_Linear:
            {
            debug_print("PSP_Linear (%d)\n", pattern_parm->pattern);
            debug_print("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmLinear));
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmLinear), cudaMemcpyHostToDevice);
#ifndef PREDEBUG            
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, 
                                            input, output, width, height);
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
           /* debug_print("PSP_Optical_Flow (%d)\n", pattern_parm->pattern);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmOpFlow));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmOpFlow), cudaMemcpyHostToDevice);*/
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
	Kernal <<<1, 1 >> >(1, 2, D_sum);


	// copy data from Device memory to host memory
	int H_sum = 0;
	error = cudaMemcpy(&H_sum, D_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	// return 
	return H_sum;
}
