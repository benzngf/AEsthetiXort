#pragma once
enum PixelSortBy {
    PSB_None,
    PSB_Hue,
    PSB_Saturation,
    PSB_Luminance,
    PSB_R,
    PSB_G,
    PSB_B,
};

enum PixelSortPattern {
    PSP_None,
    PSP_Linear,
    PSP_Radial_Spin,
    PSP_Polygon = -10,
    PSP_Spiral = 3,
    PSP_Sine,
    PSP_Triangle,
    PSP_Saw_Tooth,
    PSP_Optical_Flow,
};

#ifndef __host__
#define __host__
#endif // __host__

#ifndef __device__
#define __device__
#endif // __device__

struct Pixel {/*0.0f-255.0f*/
	union {
		struct {
			float r, g, b, a;
		};
		float e[4];
	};
	float key;
	__host__ __device__ Pixel operator+(const Pixel& rhs) const {
		Pixel result;
		result.r = r + rhs.r;
		result.g = g + rhs.g;
		result.b = b + rhs.b;
		result.a = a + rhs.a;
		result.key = key + rhs.key; // key?
		return result;
	}

	__host__ __device__ Pixel operator*(const float& coef) const {
		Pixel result;
		result.r = r * coef;
		result.g = g * coef;
		result.b = b * coef;
		result.a = a * coef;
		result.key = key * coef;
		return result;
	}
};



struct PixelSortPatternParm {
    PixelSortPattern pattern;
};

struct PixelSortPatternParmLinear {
    PixelSortPatternParm base;
    float angle;
};

struct PixelSortPatternParmRadialSpin {
    PixelSortPatternParm base;
    float center[2];
	float WHRatio;
	float rotation;
};
struct PixelSortPatternParmPolygon {
	PixelSortPatternParm base;
	float center[2];
	int numSides;
	float WHRatio;
	float rotation;
};
struct PixelSortPatternParmSpiral {
	PixelSortPatternParm base;
	float center[2];
	float curveAngle;
	float WHRatio;
	float rotation;
};
struct PixelSortPatternParmWave {//Sine, Triangle and Saw Tooth share same parm(Wave)
	PixelSortPatternParm base;
	float waveLength;
	float waveHeight;
	float rotation;
};



void PixelSortCPU(Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha);

void PixelSortGPU(Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha);

void cuLayerMemMove(Pixel* GPUinputMem_CPU, Pixel* &GPUinputMem_GPUIn, Pixel* &GPUinputMem_GPUOut, int size, int dir);
