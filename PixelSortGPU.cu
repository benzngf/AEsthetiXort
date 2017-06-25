#include "PixelSort.h"
#include <cuda.h>
#ifdef DEBUG
	#include <stdio.h>
	#include <math.h>
#endif
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define PI 3.1415926

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
__device__ __host__ __forceinline__ float angle_nomralize2(float x){
    int multiplier = (x > 0.f)? (int)(x / (2 * PI)) : (-1 + (int)(x / (2 * PI)));
    return x - 2 * PI * multiplier;
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

__device__ void RotateVector(float theta, float *in, float *out, float px, float py) {
    float x = in[0] - px;
    float y = in[1] - py;
    out[0] = px + x * cos(theta) - sin(theta) * y;
    out[1] = py + x * sin(theta) + cos(theta) * y;
}




#ifdef DEBUG
#define debug_print(...) fprintf(stderr, __VA_ARGS__) 
#else
#define debug_print(...)
#endif

#define OUPUT_POINT_MAX 5000
// r_x * r_y (x, y) + (1-r_x)* r_y (x + 1, y) + r_x * (1 - r_y) (x, y + 1) + (1-r_x)*(1-r_y) (x+1, y+1)
__device__ Pixel interpolate(const float x, const float y, 
    const float w, const float h, const Pixel* input) {

    if (x >= 0.5f && x < w - 0.5f && y >= 0.5f && y < h - 0.5f) {
        float ratiox = (x - 0.5f) - (int)(x - 0.5f);
        float ratioy = (y - 0.5f) - (int)(y - 0.5f);
        return input[(int)(y - 0.5f) * (int)w + (int)(x - 0.5f)] * (1.f-ratiox) * (1.f - ratioy)
        + input[(int)(y - 0.5f) * (int)w + (int)(x + 0.5f)] * ratiox * (1.f - ratioy)
        + input[(int)(y + 0.5f) * (int)w + (int)(x - 0.5f)] * (1.f - ratiox) * ratioy
        + input[(int)(y + 0.5f) * (int)w + (int)(x + 0.5f)] * ratiox * ratioy;
    }
    else return input[(int)y * (int)w + (int)x];
           
}

__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmRadialSpin *rspin, 
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output, bool anti_aliasing) {
    // coordinate system: regard rotation as 0 (i.e. relative coordinate system)

    float last[2], lastangle;
    int cnt = 1;


    float r = sqrtf( (x - rspin->center[0]) * (x - rspin->center[0]) + (y - rspin->center[1]) * (y - rspin->center[1]) );

    last[0] = x;
    last[1] = y;
    lastangle = angle_nomralize2(atan2f(y - rspin->center[1], x - rspin->center[0]) - rspin->rotation); // [0, 2*pi) relative to rotation
    output[0] = input[int(x) + int(y)*int(w)];

    // don't do if radius is too small
    if (r < .5f) {
        *order = 1;
        *point_cnt = cnt;
        return;
    }

    // reduce angle
    while (cnt < OUPUT_POINT_MAX) {
        lastangle -= 1.f / r;
        if (lastangle < 0.f)
            break;
        last[0] = rspin->center[0] + r * cos(lastangle + rspin->rotation);
        last[1] = rspin->center[1] + r * sin(lastangle + rspin->rotation);
        if (last[0] >= 0.f && last[0] < w && last[1] >= 0.f && last[1] < h) {
            if (input[(int)(last[0]) + (int)(last[1])*(int)(w)].key >= 0.f) {
                output[cnt] = input[(int)(last[0]) + (int)(last[1])*(int)(w)];
                cnt++;
            }
        }
        else break;
    }
    *order = cnt-1;
    //increase angle
    lastangle = angle_nomralize2(atan2f(y - rspin->center[1], x - rspin->center[0]) - rspin->rotation); // [0, 2*pi) relative to rotation
    while (cnt < OUPUT_POINT_MAX) {
        lastangle += 1.f / r;
        if (lastangle >= 2 * PI)
            break;
        last[0] = rspin->center[0] + r * cos(lastangle + rspin->rotation);
        last[1] = rspin->center[1] + r * sin(lastangle + rspin->rotation);
        if (last[0] >= 0.f && last[0] < w && last[1] >= 0.f && last[1] < h) {
            if (input[(int)(last[0]) + (int)(last[1])*(int)(w)].key >= 0.f) {
                output[cnt] = input[(int)(last[0]) + (int)(last[1])*(int)(w)];
                cnt++;
            }
            
        }
        else break;
    }
    *point_cnt = cnt;
}

__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmPolygon *polygon,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output, bool anti_aliasing) {
}

__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmSpiral *spiral,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output, bool anti_aliasing) {
}

__device__  void GetListToSortSine(
        const Pixel *input,
        PixelSortPatternParmWave *wave,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output) {
    float offset = 0.0f;
    float last[2][2];
    int cnt = 0;
    float temp_cos;
        
#define PIXELID(x, y) (int(x) + int(y)*int(w))
#ifdef SHOW_SORT
#define UPDATE_OUTPUT(x, y, id, red, green, blue) \
    output[id].r = red; \
    output[id].g = green; \
    output[id].b = blue;
#else
#define UPDATE_OUTPUT(x, y, id, red, green, blue) \
    output[cnt++] = input[id]
#endif
    int id = PIXELID(x, y);
    UPDATE_OUTPUT(x, y, id, 255, 255, 255);

    temp_cos = wave->waveHeight*cos(fmod(x - offset, wave->waveLength) / wave->waveLength * 2.0f*PI);
    last[0][0] = x + 1.0f / sqrt(1 + temp_cos*temp_cos);
    last[0][1] = y + temp_cos / sqrt(1 + temp_cos*temp_cos);
    RotateVector(wave->rotation, last[0], last[1], x, y);
    id = PIXELID(last[1][0], last[1][1]);

    while (cnt < OUPUT_POINT_MAX && 
           last[1][0] > 0 && last[1][0] < w && 
           last[1][1] > 0 && last[1][1] < h &&
           input[id].key >= 0.0f) {
        // TODO: AA here
        UPDATE_OUTPUT(last[1][0], last[1][1], id, 255, 0, 0);
        temp_cos = wave->waveHeight*cos(fmod(last[0][0] - offset, wave->waveLength) / wave->waveLength * 2.0f*PI);
        last[0][0] += 1.0f / sqrt(1 + temp_cos*temp_cos);
        last[0][1] += temp_cos / sqrt(1 + temp_cos*temp_cos);
        RotateVector(wave->rotation, last[0], last[1], x, y);
        id = PIXELID(last[1][0], last[1][1]);
    }
    
    *order = cnt-1;

    temp_cos = wave->waveHeight*cos(fmod(x - offset, wave->waveLength) / wave->waveLength * 2.0f*PI);
    last[0][0] = x - 1.0f / sqrt(1 + temp_cos*temp_cos);
    last[0][1] = y - temp_cos / sqrt(1 + temp_cos*temp_cos);
    RotateVector(wave->rotation, last[0], last[1], x, y);
    id = PIXELID(last[1][0], last[1][1]);
    while (cnt < OUPUT_POINT_MAX && 
           last[1][0] > 0 && last[1][0] < w && 
           last[1][1] > 0 && last[1][1] < h &&
           input[id].key >= 0.0f) {
        // TODO: AA here
        UPDATE_OUTPUT(last[1][0], last[1][1], id, 0, 0, 255);
        ++cnt;
        temp_cos = wave->waveHeight*cos(fmod(last[0][0] - offset, wave->waveLength) / wave->waveLength * 2.0f*PI);
        last[0][0] -= 1.0f / sqrt(1 + temp_cos*temp_cos);
        last[0][1] -= temp_cos / sqrt(1 + temp_cos*temp_cos);
        RotateVector(wave->rotation, last[0], last[1], x, y);
        id = PIXELID(last[1][0], last[1][1]);
    }
    
    *point_cnt = cnt;
#undef PIXELID
#undef UPDATE_OUTPUT
}

__device__  void GetListToSortTriangle(
        const Pixel *input,
        PixelSortPatternParmWave *wave,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output) {
}

__device__  void GetListToSortSawTooth(
        const Pixel *input,
        PixelSortPatternParmWave *wave,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output) {
}

__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmWave *wave,
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output, bool anti_aliasing) {
    switch(wave->base.pattern) {
        case PSP_Sine:
            GetListToSortSine(input, wave, x, y, w, h, order, point_cnt, output);
            break;
        case PSP_Triangle:
            GetListToSortTriangle(input, wave, x, y, w, h, order, point_cnt, output);
            break;
        case PSP_Saw_Tooth:
            GetListToSortSawTooth(input, wave, x, y, w, h, order, point_cnt, output);
            break;
        default:

#define PIXELID(x, y) (int(x) + int(y)*int(w))
            for (int i = 0; i < (int)w; ++i)
                for (int j = 0; j < (int)h; ++j) {
                    int id = PIXELID(i, j);
                    for (int k = 0; k < 3; ++k) 
                        output[id].e[k] = 100;
                }
            break;
#undef PIXELID
    }
}

// TODO: I think these code is GPU-unfriendly
__device__  void GetListToSort(
        const Pixel *input,
        PixelSortPatternParmLinear *linear, 
        const float x, const float y, 
        const float w, const float h, 
        int *order, int *point_cnt, Pixel *output, bool anti_aliasing) {
    float delta[2], last[2];
    int cnt = 1;
    
    delta[0] = cos(linear->angle);
    delta[1] = sin(linear->angle);
    
#ifdef SHOW_SORT
#define PIXELXY(x, y) (input[int(x) + int(y)*int(w)])
#define OUTPUTXY(x, y) (output[int(x) + int(y)*int(w)])
#define UPDATE_OUTPUT(x, y, red, green, blue) \
    OUTPUTXY(x, y).r = red; \
    OUTPUTXY(x, y).g = green; \
    OUTPUTXY(x, y).b = blue; 

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
#undef OUTPUTXY
#undef UPDATE_OUTPUT
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
        //output[cnt] = PIXELXY(last[0], last[1]);
        
        if (!anti_aliasing)
            output[cnt] = PIXELXY(last[0], last[1]);
        else
            output[cnt] = interpolate(last[0], last[1], w, h, input);


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
        //output[cnt] = PIXELXY(last[0], last[1]);
        if (!anti_aliasing)
            output[cnt] = PIXELXY(last[0], last[1]);
        else
            output[cnt] = interpolate(last[0], last[1], w, h, input);
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
    const int w, const int h, bool anti_aliasing) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pixelid = y * w + x;

#ifdef SHOW_SORT
    if (x == w/2+123 && y == h/2-40)
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
        GetListToSort(input, parm, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, output, anti_aliasing);
        return;
#else
        GetListToSort(input, parm, pixelx, pixely, (float)w, (float)h, &order_gpu, &point_cnt_gpu, pixel_list_gpu, anti_aliasing);
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
        output[pixelid] = pixel_list_gpu[search_index];
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
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmLinear *)pattern_parm_gpu, input, output, width, height, anti_aliasing);
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
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmRadialSpin), cudaMemcpyHostToDevice);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmRadialSpin *)pattern_parm_gpu, input, output, width, height, anti_aliasing);
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
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmPolygon), cudaMemcpyHostToDevice);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmPolygon *)pattern_parm_gpu, input, output, width, height, anti_aliasing);
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
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmSpiral), cudaMemcpyHostToDevice);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmSpiral *)pattern_parm_gpu, input, output, width, height, anti_aliasing);
            break;
        case PSP_Sine: 
        case PSP_Triangle: 
        case PSP_Saw_Tooth:
            debug_print("PSP_Wave (%d + %d)\n", PSP_Sine, pattern_parm->pattern - PSP_Sine);
            debug_print("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            debug_print("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            debug_print("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmWave));
            cudaMemcpy(pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmWave), cudaMemcpyHostToDevice);
            SortFromList<<<gdim, bdim>>>((PixelSortPatternParmWave *)pattern_parm_gpu, input, output, width, height, anti_aliasing);
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
