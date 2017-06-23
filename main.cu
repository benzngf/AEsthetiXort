#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <tuple>
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

struct PSPNamePair {
    PixelSortPatternParm *parm;
    char *name;
};

PixelSortPattern init_sort_pattern_parm(int psp_list_cnt, PSPNamePair *psp_list, int argc, char **argv) {
    PixelSortPatternParm *parm = nullptr;

    PixelSortPattern pattern = PSP_None;
    for (int i = 0; i < psp_list_cnt; ++i) {
        if (strncmp(argv[0], psp_list[i].name, strlen(psp_list[i].name)) == 0) {
            pattern = (PixelSortPattern) i;
            break;
        }
    }

    parm = psp_list[pattern].parm;
    parm->pattern = pattern;
    switch(pattern) {
        case PSP_Linear:
            ((PixelSortPatternParmLinear *)parm)->angle = atof(argv[1]);
            break;
        case PSP_Radial_Spin:
            ((PixelSortPatternParmRadialSpin *)parm)->center[0] = atof(argv[1]);
            ((PixelSortPatternParmRadialSpin *)parm)->center[1] = atof(argv[2]);
            ((PixelSortPatternParmRadialSpin *)parm)->WHRatio = atof(argv[3]);
            ((PixelSortPatternParmRadialSpin *)parm)->rotation = atof(argv[4]);
            break;
        case PSP_Polygon:
            ((PixelSortPatternParmPolygon *)parm)->center[0] = atof(argv[1]);
            ((PixelSortPatternParmPolygon *)parm)->center[1] = atof(argv[2]);
            ((PixelSortPatternParmPolygon *)parm)->numSides = atoi(argv[3]);
            ((PixelSortPatternParmPolygon *)parm)->WHRatio = atof(argv[4]);
            ((PixelSortPatternParmPolygon *)parm)->rotation = atof(argv[5]);
            break;
        case PSP_Spiral:
            ((PixelSortPatternParmSpiral *)parm)->center[0] = atof(argv[1]);
            ((PixelSortPatternParmSpiral *)parm)->center[1] = atof(argv[2]);
            ((PixelSortPatternParmSpiral *)parm)->curveAngle = atof(argv[3]);
            ((PixelSortPatternParmSpiral *)parm)->WHRatio = atof(argv[4]);
            ((PixelSortPatternParmSpiral *)parm)->rotation = atof(argv[5]);
            break;
        case PSP_Sine:
            ((PixelSortPatternParmWave *)parm)->waveLength = atof(argv[1]);
            ((PixelSortPatternParmWave *)parm)->waveHeight = atof(argv[2]);
            ((PixelSortPatternParmWave *)parm)->rotation = atof(argv[3]);
            break;
        case PSP_Triangle:
            ((PixelSortPatternParmWave *)parm)->waveLength = atof(argv[1]);
            ((PixelSortPatternParmWave *)parm)->waveHeight = atof(argv[2]);
            ((PixelSortPatternParmWave *)parm)->rotation = atof(argv[3]);
            break;
        case PSP_Saw_Tooth:
            ((PixelSortPatternParmWave *)parm)->waveLength = atof(argv[1]);
            ((PixelSortPatternParmWave *)parm)->waveHeight = atof(argv[2]);
            ((PixelSortPatternParmWave *)parm)->rotation = atof(argv[3]);
            break;
        case PSP_Optical_Flow:
            break;
        default:
            printf("Wrong pattern name: %s\n", argv[0]);
            exit(-1);
            break;
    }

    return parm->pattern;
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


    MemoryBuffer<PixelSortPatternParmLinear> psp_linear(1);
    MemoryBuffer<PixelSortPatternParmRadialSpin> psp_radial_spin(1);
    MemoryBuffer<PixelSortPatternParmPolygon> psp_polygon(1);
    MemoryBuffer<PixelSortPatternParmSpiral> psp_spiral(1);
    MemoryBuffer<PixelSortPatternParmWave> psp_wave(1);
    MemoryBuffer<PixelSortPatternParmOpFlow> psp_opflow(1);

    auto psp_linear_s = psp_linear.CreateSync(1);
    auto psp_radial_spin_s = psp_radial_spin.CreateSync(1);
    auto psp_polygon_s = psp_polygon.CreateSync(1);
    auto psp_spiral_s = psp_spiral.CreateSync(1);
    auto psp_wave_s = psp_wave.CreateSync(1);
    auto psp_opflow_s = psp_opflow.CreateSync(1);

    // FIXME: I am a bad guy, if I wrote this way, I have to check every time when I
    // add new patterns
    PSPNamePair psp_list[] = {
        { nullptr, (char *)"None" },
        { &psp_linear_s.get_cpu_rw()->base, (char *)"Linear" },
        { &psp_radial_spin_s.get_cpu_rw()->base, (char *)"Radial_Spin" },
        { &psp_polygon_s.get_cpu_rw()->base, (char *)"Polygon" },
        { &psp_spiral_s.get_cpu_rw()->base, (char *)"Spiral" },
        { &psp_wave_s.get_cpu_rw()->base, (char *)"Wave_Sine" },
        { &psp_wave_s.get_cpu_rw()->base, (char *)"Wave_Triangle" },
        { &psp_wave_s.get_cpu_rw()->base, (char *)"Wave_Saw_Tooth" },
        { &psp_opflow_s.get_cpu_rw()->base, (char *)"Optical_Flow" },
    };

    PixelSortPattern pattern = init_sort_pattern_parm(sizeof(psp_list), psp_list, argc - 9, argv + 9);

    // FIXME: I am a bad guy, if I wrote this way, I have to check every time when I
    // add new patterns
    PixelSortPatternParm *psp_list_gpu[] = {
        nullptr,
        &psp_linear_s.get_gpu_wo()->base,
        &psp_radial_spin_s.get_gpu_wo()->base,
        &psp_polygon_s.get_gpu_wo()->base,
        &psp_spiral_s.get_gpu_wo()->base,
        &psp_wave_s.get_gpu_wo()->base,
        &psp_wave_s.get_gpu_wo()->base,
        &psp_wave_s.get_gpu_wo()->base,
        &psp_opflow_s.get_gpu_wo()->base,
    };

    // NOTE: For cpu test
    //PixelSortPatternParm *pattern_parm = psp_list[pattern].parm;
    PixelSortPatternParm *pattern_parm = psp_list_gpu[pattern];

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
    for (int i = 0; i < SIZEB; ++i) {
        for (int j = 0; j < 3; ++j) {
            background_cpu[i].e[j] = imgb.get()[i*3+j];
        }
        background_cpu[i].e[3] = 0.0f;
    }
	//copy(imgb.get(), imgb.get()+SIZEB, background_cpu);

    PixelSortGPU(
            background_s.get_gpu_ro(), 
            wb, hb, 
            output_s.get_gpu_wo(), 
            sort_by, threshold_min, threshold_max,
            reverse_sort_order, pattern_parm, 
            anti_aliasing, sort_alpha);

    for (int i = 0; i < SIZEB; ++i) {
        const Pixel *a = output_s.get_cpu_ro() + i;
        const Pixel *b = background_s.get_cpu_ro() + i;
        for (int j = 0; j < 4; ++j) {
            if (a->e[j] != b->e[j]) {
                printf("%dc%d: %f != %f\n", i, j, a[j], b[j]);
                exit(-1);
            }
        }
    }

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
