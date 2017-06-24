#include "PixelSort_AE.h"


void PixelSortCPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {}



PF_Err prepareParams(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PixelSortPatternParm** out_patternparam) {
	switch (params[UIP_SortPattern]->u.pd.value) {
	case PSP_Linear: {
		PixelSortPatternParmLinear* temp = (PixelSortPatternParmLinear*)malloc(sizeof(PixelSortPatternParmLinear));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Linear;
		temp->angle = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Radial_Spin: {
		PixelSortPatternParmRadialSpin* temp = (PixelSortPatternParmRadialSpin*)malloc(sizeof(PixelSortPatternParmRadialSpin));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = (PixelSortPattern)PSP_Radial_Spin;
		temp->center[0] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.x_value)*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->center[1] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.y_value)*in_data->downsample_y.num / in_data->downsample_y.den);
		temp->WHRatio = (float)(params[UIP_WHRatio]->u.fs_d.value);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Polygon: {
		PixelSortPatternParmPolygon* temp = (PixelSortPatternParmPolygon*)malloc(sizeof(PixelSortPatternParmPolygon));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Polygon;
		temp->center[0] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.x_value)*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->center[1] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.y_value)*in_data->downsample_y.num / in_data->downsample_y.den);
		temp->numSides = (int)(params[UIP_NumSides]->u.fs_d.value);
		temp->WHRatio = (float)(params[UIP_WHRatio]->u.fs_d.value);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Spiral: {
		PixelSortPatternParmSpiral* temp = (PixelSortPatternParmSpiral*)malloc(sizeof(PixelSortPatternParmSpiral));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Spiral;
		temp->center[0] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.x_value)*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->center[1] = (float)(FIX_2_FLOAT(params[UIP_CenterPoint]->u.td.y_value)*in_data->downsample_y.num / in_data->downsample_y.den);
		temp->WHRatio = (float)(params[UIP_WHRatio]->u.fs_d.value);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Sine: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)malloc(sizeof(PixelSortPatternParmWave));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Sine;
		temp->waveLength = (float)(params[UIP_WaveLength]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->waveHeight = (float)(params[UIP_WaveHeight]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Triangle: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)malloc(sizeof(PixelSortPatternParmWave));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Triangle;
		temp->waveLength = (float)(params[UIP_WaveLength]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->waveHeight = (float)(params[UIP_WaveHeight]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Saw_Tooth: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)malloc(sizeof(PixelSortPatternParmWave));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Saw_Tooth;
		temp->waveLength = (float)(params[UIP_WaveLength]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->waveHeight = (float)(params[UIP_WaveHeight]->u.fs_d.value*in_data->downsample_x.num / in_data->downsample_x.den);
		temp->rotation = (float)(FIX_2_FLOAT(params[UIP_Angle]->u.ad.value)* PF_RAD_PER_DEGREE);
		break; }
	case PSP_Optical_Flow: {
		PixelSortPatternParmOpFlow* temp = (PixelSortPatternParmOpFlow*)malloc(sizeof(PixelSortPatternParmOpFlow));
		*out_patternparam = (PixelSortPatternParm*)temp;
		temp->base.pattern = PSP_Optical_Flow;
		break; }
	default:
		break;
	}
	return PF_Err_NONE;
}
PF_Err disposeParams(PixelSortPatternParm** out_patternparam) {
	switch ((*out_patternparam)->pattern) {
	case PSP_Linear: {
		PixelSortPatternParmLinear* temp = (PixelSortPatternParmLinear*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Radial_Spin: {
		PixelSortPatternParmRadialSpin* temp = (PixelSortPatternParmRadialSpin*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Polygon: {
		PixelSortPatternParmPolygon* temp = (PixelSortPatternParmPolygon*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Spiral: {
		PixelSortPatternParmSpiral* temp = (PixelSortPatternParmSpiral*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Sine: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Triangle: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Saw_Tooth: {
		PixelSortPatternParmWave* temp = (PixelSortPatternParmWave*)*out_patternparam;
		free(temp);
		break; }
	case PSP_Optical_Flow: {
		PixelSortPatternParmOpFlow* temp = (PixelSortPatternParmOpFlow*)*out_patternparam;
		free(temp);
		break; }
	default:
		break;
	}
	return PF_Err_NONE;
}
typedef struct ItData {
	Pixel* footage;
	int w;
	int h;
} itData;

PF_Err iterate8_toGPU(void* refcon, A_long x, A_long y, PF_Pixel *in, PF_Pixel *out) {
	itData* it = (itData*)refcon;
	if (x < 0 || y < 0 || x >= it->w || y >= it->h) {
		return PF_Err_NONE;
	}else {
		(it->footage)[y*it->w + x].r = (float) (in->red);
		(it->footage)[y*it->w + x].g = (float)(in->green);
		(it->footage)[y*it->w + x].b = (float)(in->blue);
		(it->footage)[y*it->w + x].a = (float)(in->alpha);
	}
	return PF_Err_NONE;
}
PF_Err iterate8_fromGPU(void* refcon, A_long x, A_long y, PF_Pixel *in, PF_Pixel *out) {
	itData* it = (itData*)refcon;
	if (x < 0 || y < 0 || x >= it->w || y >= it->h) {
		return PF_Err_NONE;
	}
	else {
		(out->red) = (A_u_char)(it->footage)[y*it->w + x].r;
		(out->green) = (A_u_char)(it->footage)[y*it->w + x].g;
		(out->blue) = (A_u_char)(it->footage)[y*it->w + x].b;
		(out->alpha) = (A_u_char)(it->footage)[y*it->w + x].a;
	}
	return PF_Err_NONE;
}


PF_Err prepareGPUinput(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], Pixel** &GPUinputH) {
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	itData itInput;
	itInput.w = (in_data->width*in_data->downsample_x.num / in_data->downsample_x.den);
	itInput.h = (in_data->height*in_data->downsample_y.num / in_data->downsample_y.den);
	GPUinputH = (Pixel**)suites.HandleSuite1()->host_new_handle(sizeof(Pixel) * itInput.w * itInput.h);
	itInput.footage = *GPUinputH;
	//fill the pixels
	suites.Iterate8Suite1()->iterate(in_data, 0, 0, &(params[0]->u.ld), NULL, &itInput, &iterate8_toGPU, &(params[0]->u.ld));
	return PF_Err_NONE;
}
PF_Err copyGPUresult(PF_InData *in_data, PF_OutData *out_data, PF_LayerDef *output, Pixel** GPUinputH) {
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	itData itInput;
	itInput.w = (in_data->width*in_data->downsample_x.num / in_data->downsample_x.den);
	itInput.h = (in_data->height*in_data->downsample_y.num / in_data->downsample_y.den);
	itInput.footage = *GPUinputH;
	//fill the pixels
	suites.Iterate8Suite1()->iterate(in_data, 0, 0, NULL, NULL, &itInput, &iterate8_fromGPU, output);
	suites.HandleSuite1()->host_dispose_handle((PF_Handle)GPUinputH);
	return PF_Err_NONE;
}

