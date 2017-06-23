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
    PSP_Polygon,
    PSP_Spiral,
    PSP_Sine,
    PSP_Triangle,
    PSP_Saw_Tooth,
    PSP_Optical_Flow,
};

struct Pixel {/*0.0f-255.0f*/
    union {
        struct {
            float r, g, b, a;
        };
        float e[4];
    };
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
struct PixelSortPatternParmOpFlow {
	PixelSortPatternParm base;
	//TBD
};


void PixelSortCPU(const Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha);

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha);
