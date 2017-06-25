#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "SyncedMemory.h"
#include "pgm.h"
#include "PixelSort.h"
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

// Hack
PixelSortPatternParmLinear psp_linear;
PixelSortPatternParmRadialSpin psp_radial_spin;
PixelSortPatternParmPolygon psp_polygon;
PixelSortPatternParmSpiral psp_spiral;
PixelSortPatternParmWave psp_wave;
//PixelSortPatternParmOpFlow psp_opflow;
PixelSortPatternParm * set_sort_pattern(int argc, char **argv) {
    PixelSortPatternParm *parm = nullptr;

    const char *pattern_name[] = {
        "None",
        "Linear",
        "Radial_Spin",
        "Polygon",
        "Spiral",
        "Wave_Sine",
        "Wave_Triangle",
        "Wave_Saw_Tooth",
        "Optical_Flow",
    };

    PixelSortPattern pattern = PSP_None;
    int pattern_cnt = sizeof(pattern_name)/sizeof(pattern_name[0]);
    for (int i = 0; i < pattern_cnt; ++i) {
        if (strncmp(argv[0], pattern_name[i], strlen(pattern_name[i])) == 0) {
            pattern = (PixelSortPattern) i;
            break;
        }
    }

    switch(pattern) {
        case PSP_Linear:
            psp_linear.angle = atof(argv[1]);
            parm = &psp_linear.base;
            break;
        case PSP_Radial_Spin:
            psp_radial_spin.center[0] = atof(argv[1]);
            psp_radial_spin.center[1] = atof(argv[2]);
            psp_radial_spin.WHRatio = atof(argv[3]);
            psp_radial_spin.rotation = atof(argv[4]);
            parm = &psp_radial_spin.base;
            break;
        case PSP_Polygon:
            psp_polygon.center[0] = atof(argv[1]);
            psp_polygon.center[1] = atof(argv[2]);
            psp_polygon.numSides = atoi(argv[3]);
            psp_polygon.WHRatio = atof(argv[4]);
            psp_polygon.rotation = atof(argv[5]);
            parm = &psp_polygon.base;
            break;
        case PSP_Spiral:
            psp_spiral.center[0] = atof(argv[1]);
            psp_spiral.center[1] = atof(argv[2]);
            psp_spiral.curveAngle = atof(argv[3]);
            psp_spiral.WHRatio = atof(argv[4]);
            psp_spiral.rotation = atof(argv[5]);
            parm = &psp_spiral.base;
            break;
        case PSP_Sine:
            psp_wave.waveLength = atof(argv[1]);
            psp_wave.waveHeight = atof(argv[2]);
            psp_wave.rotation = atof(argv[3]);
            parm = &psp_wave.base;
            break;
        case PSP_Triangle:
            psp_wave.waveLength = atof(argv[1]);
            psp_wave.waveHeight = atof(argv[2]);
            psp_wave.rotation = atof(argv[3]);
            parm = &psp_wave.base;
            break;
        case PSP_Saw_Tooth:
            psp_wave.waveLength = atof(argv[1]);
            psp_wave.waveHeight = atof(argv[2]);
            psp_wave.rotation = atof(argv[3]);
            parm = &psp_wave.base;
            break;
            /*
        case PSP_Optical_Flow:
            parm = &psp_opflow.base;
            break;
            */
        default:
            printf("Wrong pattern name: %s\n", argv[0]);
            exit(-1);
            break;
    }

    parm->pattern = pattern;
    return parm;
}

PixelSortBy stoPSB(char *str) {
    const char *sort_by_name[] = {
        "None",
        "Hue",
        "Saturation",
        "Luminance",
        "R",
        "G",
        "B",
    };
    
    PixelSortBy sort_by = PSB_None;
    int sort_by_cnt = sizeof(sort_by_name)/sizeof(sort_by_name[0]);
    for (int i = 0; i < sort_by_cnt; ++i) {
        if (strncmp(str, sort_by_name[i], strlen(sort_by_name[i])) == 0) {
            sort_by = (PixelSortBy) i;
            break;
        }
    }

    return sort_by;
}

int main(int argc, char **argv) {
	if (argc < 10) {
		printf("Usage: %s <input_img> <sort_by> <threshold_min> <threshold_max> <reverse_sort_order> <anti_aliasing> <sort_alpha> <output_file> <pattern_name> [pattern_parm] \n", argv[0]);
        exit(-1);
	}
    
    char *input_img = argv[1];
    PixelSortBy sort_by = stoPSB(argv[2]);
    float threshold_min = atof(argv[3]);
    float threshold_max = atof(argv[4]);
    bool reverse_sort_order = (bool)atoi(argv[5]);
    bool anti_aliasing = (bool)atoi(argv[6]);
    bool sort_alpha = (bool)atoi(argv[7]);
    char *output_file = argv[8];

    PixelSortPatternParm *pattern_parm = set_sort_pattern(argc - 9, argv + 9);


	bool sucb;
	int wb, hb, cb;
	auto imgb = ReadNetpbm(wb, hb, cb, sucb, input_img);
	if (!sucb) {
		puts("Something wrong with reading the input image files.");
        exit(-1);
	}
	if (cb != 3) {
		puts("The background and target image must be colored.");
        exit(-1);
	}

	const int SIZEB = wb*hb;
	MemoryBuffer<Pixel> background(SIZEB), output(SIZEB);
	auto background_s = background.CreateSync(SIZEB);
	auto output_s = output.CreateSync(SIZEB);

	Pixel *background_cpu = background_s.get_cpu_wo();
    Pixel *output_cpu = output_s.get_cpu_wo();
    for (int i = 0; i < SIZEB; ++i) {
        for (int j = 0; j < 3; ++j) {
            background_cpu[i].e[j] = imgb.get()[i*3+j];
            output_cpu[i].e[j] = 0;
        }
        output_cpu[i].e[3] = 255.0f;
        background_cpu[i].e[3] = 255.0f;
    }
	//copy(imgb.get(), imgb.get()+SIZEB, background_cpu);

    PixelSortGPU(
            background_s.get_gpu_rw(), 
            wb, hb, 
            output_s.get_gpu_rw(), 
            sort_by, threshold_min, threshold_max,
            reverse_sort_order, pattern_parm, 
            anti_aliasing, sort_alpha);


	unique_ptr<uint8_t[]> o(new uint8_t[SIZEB*3]);
	const Pixel *o_cpu_pixel = output_s.get_cpu_ro();
    float *o_cpu = (float *)malloc(sizeof(float)*SIZEB*3);
    for (int i = 0; i < SIZEB; ++i) {
        for (int j = 0; j < 3; ++j) {
            o_cpu[i*3+j] = o_cpu_pixel[i].e[j];
        }
    }
	//const float *o_cpu = output_s.get_cpu_ro();
	transform(o_cpu, o_cpu+SIZEB*3, o.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });
	WritePPM(o.get(), wb, hb, output_file);
	return 0;
}
