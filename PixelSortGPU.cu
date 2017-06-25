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
__device__ __host__ __forceinline__ float MaxRGB(const float R, const float G, const float B) {
    if(R > G && R > B) return R;
    else if (G > R && G > B) return G;
    else if (B > R && B > G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ __forceinline__ float MinRGB(const float R, const float G, const float B) {
    if(R < G && R < B) return R;
    else if (G < R && G < B) return G;
    else if (B < R && B < G) return B;
    else return R; // R == G && R == B
}
__device__ __host__ __forceinline__ float absolute(float x){
    if (x >= 0)
        return x;
    return -x;
}
__device__ __host__ __forceinline__ float angle_nomralize(float x){
    if (x < 0.0)
        return x + 6.0;
    return x;
}
__device__ __host__ float getLuminance(const float R, const float G, const float B) {
    return (MaxRGB(R, G, B) + MinRGB(R, G, B)) / 510.f;
}  
__device__ __host__ float getHue(const float R, const float G, const float B) {
    float M = MaxRGB(R, G, B);
    float m = MinRGB(R, G, B);
    float C = M - m;

    float Result = 0.0f;
    if (C > -0.1f && C < 0.1f) //C == 0.0f
        Result = 0.0f;
    else if (M == R)
        Result = angle_nomralize((G - B) / C);
    else if (M == G)
        Result = 2.0f + ( (B - R) / C );
    else if (M == B)
        Result = 4.0f + ( (R - G) / C );
    return 60.0f*Result;
}
__device__ __host__ float getSaturation(const float R, const float G, const float B) {
    float C = (MaxRGB(R, G, B) - MinRGB(R, G, B)) / 255.f;
    float L = getLuminance(R, G, B);

    if (C == 0.f) return 0.f;
    else if (L <= 0.5f) return (C > (2.f * L))? 1.0 : (C / (2.f * L));
    else return (C > (2.f - 2.f * L))? 1.0 : C / (2.f - 2.f * L);
}
// For the following 3 functions, adapted from: https://www.jiuzhang.com/solutions/kth-largest-element/
__device__ __host__ int kthLargestPartition(int l, int r, Pixel pixel_array[]) {

    int left = l, right = r;
    Pixel temp = pixel_array[left];
    float pivot = temp.key;
       

    while (left < right) {
        while (left < right && pixel_array[right].key >= pivot) {
            right--;
        }
        pixel_array[left] = pixel_array[right];
        while (left < right && pixel_array[left].key <= pivot) {
           left++;
        }
        pixel_array[right] = pixel_array[left];
    }        

    pixel_array[left] = temp;

    return left;         

}
__device__ __host__ int kthLargestInternal(int l, int r, int k, Pixel pixel_array[]) {
    if (l == r)
        return l;

    int position = kthLargestPartition(l, r, pixel_array);
    if (position + 1 == k)
        return position;
    else if (position + 1 < k)
        return kthLargestInternal(position + 1, r, k, pixel_array);
    else
        return kthLargestInternal(l, position - 1, k, pixel_array);
}
__device__ __host__ int kthLargest(int k, int length, Pixel pixel_array[]) {
    if (length == 0 || k <= 0)
        return -1;
    return kthLargestInternal(0, length - 1, length - k + 1, pixel_array);
}  


#ifdef DEBUG
#define debug_print(...) fprintf(stderr, __VA_ARGS__) 
#else
#define debug_print(...)
#endif

#define OUPUT_POINT_MAX 5000


// TODO: I think these code is GPU-unfriendly
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
    
#ifdef SHOW_SORT
#define PIXELXY(x, y) (input[int(x) + int(y)*int(w)])
#define OUTPUTXY(x, y) (output[int(x) + int(y)*int(w)])
#define UPDATE_OUTPUT(x, y, red, green, blue) \
    OUTPUTXY(x, y).r = PIXELXY(x, y).r; \
    OUTPUTXY(x, y).g = PIXELXY(x, y).g; \
    OUTPUTXY(x, y).b = PIXELXY(x, y).b; 

    UPDATE_OUTPUT(x, y, 255, 255, 255);
    
    // prev
    last[0] = x - delta[0];
    last[1] = y - delta[1];
    while (cnt < OUPUT_POINT_MAX && 
           last[0] > 0 && last[0] < w && 
           last[1] > 0 && last[1] < h &&
           PIXELXY(last[0], last[1]).key >= 0.0f) {
        // TODO: AA here
        UPDATE_OUTPUT(last[0], last[1], 255, 0, 0);
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
           PIXELXY(last[0], last[1]).key >= 0.0f) {
        // TODO: AA here
        UPDATE_OUTPUT(last[0], last[1], 0, 0, 255);
        ++cnt;
        last[0] += delta[0];
        last[1] += delta[1];
    }
#undef PIXELXY
#else
#define PIXELXY(x, y) (input[int(x) + int(y)*int(w)])
    output[0] = PIXELXY(x, y);
    
    // prev
    last[0] = x - delta[0];
    last[1] = y - delta[1];
    while (cnt < OUPUT_POINT_MAX && 
           last[0] > 0 && last[0] < w && 
           last[1] > 0 && last[1] < h &&
           PIXELXY(last[0], last[1]).key >= 0.0f) {
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
           PIXELXY(last[0], last[1]).key >= 0.0f) {
        // TODO: AA here
        output[cnt] = PIXELXY(last[0], last[1]);
        ++cnt;
        last[0] += delta[0];
        last[1] += delta[1];
    }
#undef PIXELXY
#endif

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
#ifdef SHOW_SELECT
        Pixel *inout, Pixel *output)
#else
        Pixel *inout)
#endif
{
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
        /*
        if (cur->key > 0.0f) {
            cur->r = 255.0f;
            cur->g /= 2;
            cur->b /= 2;
        }
        */


#ifdef SHOW_SELECT
        if (cur->key >= 0.0f) {
            output[pixelid].r = output[pixelid].g = output[pixelid].b = 255.0f;
        }
#endif
    }
}

/*(input image, image width, image height, output image(to fill),
sort by? (RGB...), threshold_min, threshold, max, reverse?
pattern parameter, do antialiasing?, sort alpha?)*/

template <typename Parm>
__global__ void SortFromList(Parm *parm, 
    const Pixel *input, Pixel *output, 
    const int w, const int h) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelid = y * w + x;

#ifdef SHOW_SORT
    if (x == w/2 && y == h/2)
#else
    if (x < w && y < h)
#endif
    {
        const float pixelx = x + 0.5;
        const float pixely = y + 0.5;

        int point_cnt_gpu;
        int order_gpu;


        Pixel pixel_list_gpu[OUPUT_POINT_MAX];

#ifdef SHOW_SORT
        GetListToSort(input, parm, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, output);
        return;
#else
        GetListToSort(input, parm, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, pixel_list_gpu);
#endif
        // Sorting


#ifndef SORT_TEST

        int search_index = kthLargest(point_cnt_gpu - order_gpu, point_cnt_gpu, pixel_list_gpu);
#else
        for (int i = 0; i < point_cnt_gpu; i++) {
            for (int j = 0; j < point_cnt_gpu - i; j++){
                if (pixel_list_gpu[j].key > pixel_list_gpu[j+1].key) {
                    Pixel temp = pixel_list_gpu[j];
                    pixel_list_gpu[j] = pixel_list_gpu[j+1];
                    pixel_list_gpu[j+1] = temp;
                }
            }
        }
        int search_index = order_gpu;
#endif

        // Fill value: order
        output[pixelid].r = pixel_list_gpu[search_index].r;
        output[pixelid].g = pixel_list_gpu[search_index].g;
        output[pixelid].b = pixel_list_gpu[search_index].b;
        output[pixelid].a = pixel_list_gpu[search_index].a;
    }
}

void PixelSortGPU(Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

    dim3 gdim(CeilDiv(width, 32), CeilDiv(height, 16)), bdim(32, 16);

#ifdef SHOW_SELECT
    ComputeKey<<<gdim, bdim>>>(sort_by, width, height, threshold_min, threshold_max, input, output);
    return;
#else
    ComputeKey<<<gdim, bdim>>>(sort_by, width, height, threshold_min, threshold_max, input);
#endif

    PixelSortPatternParm *pattern_parm_gpu = nullptr;
    switch (pattern_parm->pattern) {
        case PSP_Linear:
            {
            debug_print("PSP_Linear (%d)\n", pattern_parm->pattern);
            debug_print("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmLinear));
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmLinear), cudaMemcpyHostToDevice);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, input, output, width, height);
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

        /*
        case PSP_Optical_Flow:
            debug_print("PSP_Optical_Flow (%d)\n", pattern_parm->pattern);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmOpFlow));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmOpFlow), cudaMemcpyHostToDevice);
            break;
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

void cuLayerMemMove(Pixel* GPUinputMem_CPU, Pixel* &GPUinputMem_GPUIn, Pixel* &GPUinputMem_GPUOut, int size,int dir) {
	if (dir == 0) {
		cudaMalloc((void**)&GPUinputMem_GPUIn, sizeof(Pixel)*size);
		cudaMalloc((void**)&GPUinputMem_GPUOut, sizeof(Pixel)*size);
		cudaMemcpy(GPUinputMem_GPUIn, GPUinputMem_CPU, sizeof(Pixel)*size, cudaMemcpyHostToDevice);
	}
	else if (dir == 1) {
		cudaMemcpy(GPUinputMem_CPU, GPUinputMem_GPUOut, sizeof(Pixel)*size, cudaMemcpyDeviceToHost);
		cudaFree(GPUinputMem_GPUIn);
		cudaFree(GPUinputMem_GPUOut);
	}
}
